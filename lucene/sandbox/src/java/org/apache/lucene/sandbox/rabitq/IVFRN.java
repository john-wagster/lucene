package org.apache.lucene.sandbox.rabitq;

import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.Random;

import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

import static org.apache.lucene.util.VectorUtil.dotProduct;
import static org.apache.lucene.util.VectorUtil.scaleMaxInnerProductScore;

public class IVFRN {
  private Factor[] fac;

  private int N; // the number of data vectors
  private int C; // the number of clusters

  private int[] start; // the start point of a cluster
  private int[] len; // the length of a cluster
  private int[] id; // N of size_t the ids of the objects in a cluster
  private float[] distToC; // N of floats distance to the centroids (not the squared distance)
  private float[] u; // B of floats random numbers sampled from the uniform distribution [0,1]

  // FIXME: FUTURE - make this a byte[] instead??
  private byte[][] binaryCode; // (B / 8) * N of 64-bit uint64_t

  private float[] x0; // N of floats in the Random Net algorithm
  private float[][]
      centroids; // N * B floats (not N * D), note that the centroids should be randomized
  private int[] dataMapping;

  private final int B;
  private final int D;

  public IVFRN(
      int numVecs,
      float[][] centroids,
      float[] distToCentroid,
      float[] _x0,
      int[] clusterId,
      byte[][] binary,
      int dimensions)
      throws IOException {
    // FIXME: clean up all the weird offset mgmt ... store mappings instead of repacking data
    // everywhere
    // FIXME: stop serializing all of X here (in save) ... instead store a mapping
    D = dimensions;
    B = (D + 63) / 64 * 64;

    N = numVecs;
    C = centroids.length;

    // Check if B is greater than or equal to D
    assert (B >= D);

    start = new int[C];
    len = new int[C];
    id = new int[N];
    distToC = new float[N];
    x0 = new float[N];

    for (int i = 0; i < N; i++) {
      len[clusterId[i]]++;
    }
    int sum = 0;
    for (int i = 0; i < C; i++) {
      start[i] = sum;
      sum += len[i];
    }

    for (int i = 0; i < N; i++) {
      id[start[clusterId[i]]] = i;
      distToC[start[clusterId[i]]] = distToCentroid[i];
      x0[start[clusterId[i]]] = _x0[i];
      start[clusterId[i]]++;
    }

    for (int i = 0; i < C; i++) {
      start[i] -= len[i];
    }

    this.centroids = centroids;

    this.dataMapping = new int[N];
    this.binaryCode = new byte[N][B / 8];
    for (int i = 0; i < N; i++) {
      int x = id[i];
      dataMapping[i] = x;
      binaryCode[i] = binary[x];
    }
  }

  private IVFRN(
      int n,
      int d,
      int c,
      int b,
      float[][] centroids,
      int[] dataMapping,
      byte[][] binaryCode,
      int[] start,
      int[] len,
      int[] id,
      float[] distToC,
      float[] x0,
      float[] u,
      Factor[] fac) {
    this.N = n;
    this.D = d;
    this.C = c;
    this.B = b;

    this.centroids = centroids;
    this.dataMapping = dataMapping;
    this.binaryCode = binaryCode;
    this.start = start;
    this.len = len;
    this.id = id;
    this.distToC = distToC;
    this.x0 = x0;
    this.u = u;
    this.fac = fac;
  }

  public void save(String filename) throws IOException {
    try (FileOutputStream fos = new FileOutputStream(filename);
        FileChannel fc = fos.getChannel()) {
      ByteBuffer bb = ByteBuffer.allocate(4 * 4).order(ByteOrder.LITTLE_ENDIAN);
      bb.putInt(N);
      bb.putInt(D);
      bb.putInt(C);
      bb.putInt(B);
      bb.flip();
      fc.write(bb);

      // start, len, id, distToC, x0, centroids, data, binaryCode
      bb =
          ByteBuffer.allocate(4 * C + 4 * C + 4 * N + 4 * N + 4 * N).order(ByteOrder.LITTLE_ENDIAN);
      for (int i = 0; i < C; i++) {
        bb.putInt(start[i]);
      }

      for (int i = 0; i < C; i++) {
        bb.putInt(len[i]);
      }

      for (int i = 0; i < N; i++) {
        bb.putInt(id[i]);
      }

      for (int i = 0; i < N; i++) {
        bb.putFloat(distToC[i]);
      }

      for (int i = 0; i < N; i++) {
        bb.putFloat(x0[i]);
      }
      bb.flip();
      fc.write(bb);

      bb = ByteBuffer.allocate(4 * C * B).order(ByteOrder.LITTLE_ENDIAN);
      for (int i = 0; i < C; i++) {
        for (int j = 0; j < B; j++) {
          bb.putFloat(centroids[i][j]);
        }
      }
      bb.flip();
      fc.write(bb);

      bb = ByteBuffer.allocate(4 * N).order(ByteOrder.LITTLE_ENDIAN);
      for (int i = 0; i < N; i++) {
        bb.putInt(dataMapping[i]);
      }
      bb.flip();
      fc.write(bb);

      for (int i = 0; i < N; i++) {
        bb = ByteBuffer.allocate(8 * B / 64).order(ByteOrder.LITTLE_ENDIAN);
        for (int j = 0; j < B / 8; j++) {
          bb.put(binaryCode[i][j]);
        }
        bb.flip();
        fc.write(bb);
      }
    }
  }

  public static IVFRN load(String filename) throws IOException {
    try (FileInputStream fis = new FileInputStream(filename);
        FileChannel fc = fis.getChannel()) {
      ByteBuffer bb = ByteBuffer.allocate(4 * 4).order(ByteOrder.LITTLE_ENDIAN);
      fc.read(bb);
      bb.flip();
      int N = bb.getInt();
      int D = bb.getInt();
      int C = bb.getInt();
      int B = bb.getInt();

      float fac_norm = (float) Utils.constSqrt(1.0 * B);
      float max_x1 = (float) (1.9 / Utils.constSqrt(1.0 * B - 1.0));

      float[][] centroids = new float[C][B];
      int[] dataMapping = new int[N];

      byte[][] binaryCode = new byte[N][B / 8];
      int[] start = new int[C];
      int[] len = new int[C];
      int[] id = new int[N];
      float[] distToC = new float[N];
      float[] x0 = new float[N];

      // start, len, id, distToC, x0, centroids, data, binaryCode
      bb =
          ByteBuffer.allocate(4 * C + 4 * C + 4 * N + 4 * N + 4 * N).order(ByteOrder.LITTLE_ENDIAN);
      fc.read(bb);
      bb.flip();

      for (int i = 0; i < C; i++) {
        start[i] = bb.getInt();
      }

      for (int i = 0; i < C; i++) {
        len[i] = bb.getInt();
      }

      for (int i = 0; i < N; i++) {
        id[i] = bb.getInt();
      }

      for (int i = 0; i < N; i++) {
        distToC[i] = bb.getFloat();
      }

      for (int i = 0; i < N; i++) {
        x0[i] = bb.getFloat();
      }

      bb = ByteBuffer.allocate(4 * C * B).order(ByteOrder.LITTLE_ENDIAN);
      fc.read(bb);
      bb.flip();
      for (int i = 0; i < C; i++) {
        for (int j = 0; j < B; j++) {
          centroids[i][j] = bb.getFloat();
        }
      }

      bb = ByteBuffer.allocate(4 * N).order(ByteOrder.LITTLE_ENDIAN);
      fc.read(bb);
      bb.flip();
      for (int i = 0; i < N; i++) {
        dataMapping[i] = bb.getInt();
      }

      for (int i = 0; i < N; i++) {
        bb = ByteBuffer.allocate(8 * B / 64).order(ByteOrder.LITTLE_ENDIAN);
        fc.read(bb);
        bb.flip();
        for (int j = 0; j < B / 8; j++) {
          binaryCode[i][j] = bb.get();
        }
      }

      Random random = new Random(1);
      float[] u = new float[B];
      for (int i = 0; i < B; i++) {
        u[i] = (float) random.nextDouble();
      }

      // FIXME: speed up with panama
      Factor[] fac = new Factor[N];
      for (int i = 0; i < N; i++) {
        double x_x0 = distToC[i] / x0[i]; // ∥o𝑟 − c∥ / <o¯, o>
        float sqrX = distToC[i] * distToC[i];
        float error = (float) (2.0 * max_x1
                * Math.sqrt(x_x0 * x_x0 - distToC[i] * distToC[i]));
        float factorPPC = (float)
                (-2.0 / fac_norm * x_x0
                    * ((float) SpaceUtils.popcount(binaryCode[i], B) * 2.0 - B));
        float factorIP = (float) (-2.0 / fac_norm * x_x0);
        fac[i] = new Factor(sqrX, error, factorPPC, factorIP);
      }

      return new IVFRN(
          N, D, C, B, centroids, dataMapping, binaryCode, start, len, id, distToC, x0, u, fac);
    }
  }

  record QuantizedQuery(
      byte[] result, int sumQ, float centroidDist, float vl, float width, int centroidId) {}

  public int getCentroidId(int vectorNodeId) {
    int centroidPos = Arrays.binarySearch(start, vectorNodeId);
    if (centroidPos == 0) {
      return 0;
    }
    assert centroidPos < 0;
    // Flip the sign and subtract 2 to get the centroid id
    return -centroidPos - 2;
  }

  public QuantizedQuery[] quantizeQuery(float[] query) {
    QuantizedQuery[] quantizedQueries = new QuantizedQuery[C];
    for (int c = 0; c < C; c++) {
      float sqrY = VectorUtils.squareDistance(query, centroids[c]);

      // Preprocess the residual query and the quantized query
      float[] v = SpaceUtils.range(query, centroids[c]);
      float vl = v[0], vr = v[1];
      float width = (vr - vl) / ((1 << SpaceUtils.B_QUERY) - 1);

      QuantResult quantResult = SpaceUtils.quantize(query, centroids[c], u, vl, width);
      byte[] byteQuery = quantResult.result();
      int sumQ = quantResult.sumQ();

      byte[] quantQuery = SpaceUtils.transposeBinByte(byteQuery, D);
      quantizedQueries[c] = new QuantizedQuery(quantQuery, sumQ, sqrY, vl, width, c);
    }
    return quantizedQueries;
  }

  public float quantizeCompare(QuantizedQuery quantizedQuery, int nodeId) {
    int c = quantizedQuery.centroidId();
    float sqrY = quantizedQuery.centroidDist();
    float vl = quantizedQuery.vl();
    float width = quantizedQuery.width();
    byte[] quantQuery = quantizedQuery.result();
    int sumQ = quantizedQuery.sumQ();

    int startC = start[c];
    assert nodeId >= startC && nodeId < startC + len[c];

    float tmpDist = 0;
    long qcDist = SpaceUtils.ipByteBinBytePan(quantQuery, binaryCode[nodeId]);

    tmpDist +=
        fac[nodeId].sqrX()
            + sqrY
            + fac[nodeId].factorPPC() * vl
            + (qcDist * 2 - sumQ) * fac[nodeId].factorIP() * width;
    return tmpDist;
  }

  public IVFRNResult indexToIndexTest(
          RandomAccessVectorValues.Floats dataVectors, float[] query, int k, int nProbe)
          throws IOException {

    int origin = 2176;
    int[] neighbors = new int[] {2176, 3752, 882, 4009, 2837, 190, 3615, 816, 1045, 1884, 1, 2, 3};
    float[] originV = Arrays.copyOf(dataVectors.vectorValue(origin), 128);

    float[] trueDists = new float[len[0]];
    float[] estDists = new float[len[0]];
    float[] estDistsRBQ = new float[len[0]];

    for(int i = 0; i < len[0]; i++) {
//      int nextI = neighbors[i];
      int nextI = i;
      float[] nextV = Arrays.copyOf(dataVectors.vectorValue(nextI), 128);
      float trueD = VectorUtil.squareDistance(originV, nextV);
      trueDists[i] = trueD;

//      float xq = nextV.length * 8f - VectorUtil.xorBitCount(binaryCode[origin], binaryCode[nextI]);
//      float xq = nextV.length * 8f - VectorUtil.xorBitCount(binaryCode[origin], binaryCode[nextI]);
      SpaceUtils.B_QUERY = 1;

      long xq = SpaceUtils.ipByteBinBytePan(binaryCode[origin], binaryCode[nextI]);

      float[] v = SpaceUtils.range(nextV, centroids[0]);
      float vl = v[0], vr = v[1];
      // Δ := (𝑣𝑟 − 𝑣𝑙)/(2𝐵𝑞 − 1)
      float width = (vr - vl) / ((1 << SpaceUtils.B_QUERY) - 1);
      float sumQ = SpaceUtils.quantize(nextV, centroids[0], u, vl, width).sumQ();


//      (2Δ / √𝐷) * ⟨x¯𝑏, q′𝑢⟩ + (2𝑣𝑙 / √𝐷) * ∑︁𝐷𝑖=1(x¯𝑏[𝑖]) − Δ / √𝐷 * ∑︁𝐷𝑖=1(q¯𝑢 [𝑖]) − √𝐷 · 𝑣𝑙
      float rbqEstOV = fac[nextI].sqrX() + fac[origin].sqrX() + fac[nextI].factorPPC() * vl + (xq * 2 - sumQ) * fac[nextI].factorIP() * width;

      v = SpaceUtils.range(originV, centroids[0]);
      vl = v[0];
      vr = v[1];
      // Δ := (𝑣𝑟 − 𝑣𝑙)/(2𝐵𝑞 − 1)
      width = (vr - vl) / ((1 << SpaceUtils.B_QUERY) - 1);
      sumQ = SpaceUtils.quantize(originV, centroids[0], u, vl, width).sumQ();

      float rbqEstVO = fac[nextI].sqrX() + fac[origin].sqrX() + fac[origin].factorPPC() * vl + (xq * 2 - sumQ) * fac[origin].factorIP() * width;

      long xor = VectorUtil.xorBitCount(binaryCode[origin], binaryCode[nextI]);
      float estimatorBaseline = -(nextV.length * 8f - xor);

      float estimator1 = (xor * 2 - nextV.length * 8f) / (nextV.length * 8f);
      float estimator2 = fac[nextI].sqrX() + fac[origin].sqrX() + fac[origin].factorPPC() * vl + (xor-B) * fac[origin].factorIP() * width;
      float estimator3 = fac[nextI].sqrX() + fac[origin].sqrX() + fac[origin].factorPPC() * vl + (xor*2-sumQ) * fac[origin].factorIP() * width;

      float[] C = centroids[0];
      float Cnorm = norm(C);

      estDists[i] = estimatorBaseline;
      estDistsRBQ[i] = rbqEstOV;

      System.out.println("foo");
    }

    try (BufferedWriter outputWriter = new BufferedWriter(new FileWriter("voest.out"));) {
      outputWriter.write(Arrays.toString(estDists));
    }

    try (BufferedWriter outputWriter = new BufferedWriter(new FileWriter("voestrbq.out"));) {
      outputWriter.write(Arrays.toString(estDistsRBQ));
    }

    try (BufferedWriter outputWriter = new BufferedWriter(new FileWriter("votruth.out"));) {
      outputWriter.write(Arrays.toString(trueDists));
    }

    IVFRNStats stats =
            new IVFRNStats(
                    1,
                    1,
                    1,
                    1 / 1);
    return new IVFRNResult(null, stats);
  }

    public IVFRNResult search(
      RandomAccessVectorValues.Floats dataVectors, float[] query, int k, int nProbe)
      throws IOException {
    // FIXME: FUTURE - implement fast scan and do a comparison

    assert nProbe < C;
//    float distK = Float.MAX_VALUE;
    float distK = Float.MIN_VALUE;
//    PriorityQueue<Result> knns = new PriorityQueue<>(k, Comparator.reverseOrder());
    PriorityQueue<Result> knns = new PriorityQueue<>(k);

    // Find out the nearest N_{probe} centroids to the query vector.
    PriorityQueue<Result> topNProbeCentroids = new PriorityQueue<>(nProbe);
    for (int i = 0; i < C; i++) {
      topNProbeCentroids.add(new Result(VectorUtils.squareDistance(query, centroids[i]), i));
    }

    Result[] centroidDist = new Result[C];
    for (int i = 0; i < nProbe; i++) {
      centroidDist[i] = topNProbeCentroids.remove();
    }

    // FIXME: FUTURE - don't use the Result class for this; it's confusing
    // FIXME: FUTURE - hardcoded
    int maxEstimatorSize = 500;
//    PriorityQueue<Result> estimatorDistances =
//        new PriorityQueue<>(maxEstimatorSize, Comparator.reverseOrder());
    PriorityQueue<Result> estimatorDistances =
            new PriorityQueue<>(maxEstimatorSize);

    float errorBoundAvg = 0f;
    int errorBoundTotalCalcs = 0;
    int totalEstimatorQueueAdds = 0;
    int floatingPointOps = 0;
    for (int pb = 0; pb < nProbe; pb++) {
      int c = centroidDist[pb].c();
      float sqrY = centroidDist[pb].sqrY();

      if (!Float.isFinite(sqrY)) {
        continue;
      }

      // Preprocess the residual query and the quantized query
      float[] v = SpaceUtils.range(query, centroids[c]);
      float vl = v[0], vr = v[1];
      // Δ := (𝑣𝑟 − 𝑣𝑙)/(2𝐵𝑞 − 1)
      float width = (vr - vl) / ((1 << SpaceUtils.B_QUERY) - 1);

      //////////////
//      float[] normQuery = new float[query.length];
//      float norm = (float) calculateMagnitude(query);
//      for(int i = 0; i < query.length; i++) {
//        normQuery[i] = query[i] / norm;
//      }
      ///////////////

      // query = q′
      QuantResult quantResult = SpaceUtils.quantize(query, centroids[c], u, vl, width);
      // q¯ = Δ · q¯𝑢 + 𝑣𝑙 · 1𝐷
      // q¯ is an approximation of q′  (scalar quantized approximation)
      byte[] byteQuery = quantResult.result();
      int sumQ = quantResult.sumQ();

      // Binary String Representation
//      byte[] quantQuery = SpaceUtils.transposeBin(byteQuery, D);
      byte[] quantQuery = SpaceUtils.transposeBinByte(byteQuery, D);

      int startC = start[c];
      float y = (float) Math.sqrt(sqrY);

      int facCounter = startC;
      int bCounter = startC;


      // TEMPORARY FACTORS - can precompute several of these
      float[] C = centroids[c];
      float QC = norm(subtract(query, C));
      float[] QmC = subtract(query, C);
      float QmCdC = VectorUtil.dotProduct(QmC, C);
      float Cnorm = norm(C);
      float QdC = VectorUtil.dotProduct(query, C);
      float Qnorm = norm(query);
      float[] QdQnorm = divide(query, Qnorm);

      float avgDiff = 0f;
      float[] tmpdists = new float[len[c]];
      float[] rbqs = new float[len[c]];
      float[] estimators = new float[len[c]];
      float[] reals = new float[len[c]];
      float[] rbqEsts = new float[len[c]];

      for (int i = 0; i < len[c]; i++) {
        // ⟨x¯𝑏, q𝑢¯(𝑗)⟩
        long qcDist = SpaceUtils.ipByteBinBytePan(quantQuery, binaryCode[bCounter]);

        // ∥o𝑟 − c∥^2
        float OrC2 = fac[facCounter].sqrX();

        // ∥q𝑟 − c∥^2
        float QrC = y;
        float QrC2 = sqrY;

        //// Paper Formulas
        //∥o𝑟 − q𝑟∥^2 = ∥o𝑟 − c∥^2 + ∥q𝑟 − c∥^2 − 2·∥o𝑟 − c∥·∥q𝑟 − c∥·⟨q, o⟩
        // estimator of ⟨o, q⟩ = ⟨o¯,q⟩ / ⟨o¯,o⟩ - errorBound
        // where
        // ⟨o¯, q⟩ = ⟨x¯, q′⟩ and ⟨x¯, q¯⟩ is a scalar quantized approximation
        // ⟨x¯, q¯⟩ = ⟨(2x¯𝑏 − 1𝐷) / √𝐷, Δ · q¯𝑢 + 𝑣𝑙 · 1𝐷⟩
        // ⟨x¯, q¯⟩ = (2Δ / √𝐷) * ⟨x¯𝑏, q′𝑢⟩ + (2𝑣𝑙 / √𝐷) * ∑︁𝐷𝑖=1(x¯𝑏[𝑖]) − Δ / √𝐷 * ∑︁𝐷𝑖=1(q¯𝑢 [𝑖]) − √𝐷 · 𝑣𝑙
        // ⟨x¯𝑏, q¯𝑢⟩ = ∑︁𝐵q-1𝑗=0(2𝑗·⟨x¯𝑏, q𝑢¯(𝑗)⟩
        // errorBound = √︄((1 − ⟨o¯, o⟩^2) / ⟨o¯, o⟩^2) * (𝜖0 / √(𝐷 − 1))
        ////

        // ORIGINAL RBQ estimator
//         float rbqEst = OrC2 + QrC2 + fac[facCounter].factorPPC() * vl + (qcDist * 2 - sumQ) * fac[facCounter].factorIP() * width;

        // ALT RBQ factor estimator
//        float rbqEst = OrC2 + QrC2 + fac[facCounter].factorPPC() * vl + (qcDist * 2) * fac[facCounter].factorIP() * width;

        // ALT RBQ factor estimator
        float rbqEst = fac[facCounter].factorPPC() * vl + (qcDist * 2 - sumQ) * fac[facCounter].factorIP() * width;
//        float rbqEst = OrC2 + QrC2 + fac[facCounter].factorPPC() * vl + (qcDist * 2 - sumQ) * fac[facCounter].factorIP() * width;

//        rbqEst = rbqEst * 0.001f;
//        rbqEst = rbqEst / (float) Math.pow(Cnorm, 2);
//        rbqEst = rbqEst / Cnorm; // <--- FIXME: This should be what rbqEst is???? and remove 275 below ... why doesn't that work!
//        rbqEst = rbqEst * 0.0045215f;
        rbqEst = rbqEst / (float) Math.pow(Qnorm, 2);  // <--- the correct thing to use???
//        rbqEst = rbqEst / Qnorm;
        //FIXME: get the actual factor here by dividing rbq by tmpdist & scatterplot

        // TEMPORARY FACTORS - can precompute several of these
        float[] O = dataVectors.vectorValue(dataMapping[startC]+i);
        float OC = norm(subtract(O, C));
        float OdC = VectorUtil.dotProduct(O, C);
        float[] OmC = subtract(O, C);
//        float OdQdvQnorm = VectorUtil.dotProduct(O, QdQnorm);

        // TARGET (footnote 8)
        // ⟨o, q⟩ = ∥o − c∥ · ∥q − c∥ · ⟨(o − c)/∥o − c∥, (q − c)/∥q − c∥ ⟩ + ⟨o, c⟩ + ⟨q, c⟩ − ∥c∥^2
        float rbq = VectorUtil.dotProduct(
                divide(OmC, norm(OmC)),
                divide(QmC, norm(QmC)));
        float tmpDist2 = OC * QC * rbq + OdC + QdC - (float) Math.pow(Cnorm, 2);  // 100% RECALL on 5 query vectors
        avgDiff += rbqEst / rbq;
//        System.out.print(tmpDist + ", ");
        rbqEsts[i] = rbqEst;
        rbqs[i] = rbq;
        reals[i] = tmpDist2;
        float tmpDist = OC * QC * rbqEst + OdC + QdC - (float) Math.pow(Cnorm, 2);  // ??% RECALL on 5 query vectors
//        tmpDist = 175-tmpDist+100;
        tmpDist = 2*QdC-tmpDist;
//        tmpDist = scaleMaxInnerProductScore(tmpDist);
        tmpdists[i] = tmpDist;
//        tmpDist = tmpDist2;

        // ALT 1 (gaoj0017)
        // ⟨o, q⟩ = ∥o𝑟 − c∥ · ∥q𝑟∥ · ⟨o, q𝑟 / ∥q𝑟∥⟩ + ⟨c,q𝑟⟩
//        tmpDist = OC * norm(query) * OdQdvQnorm + QdC; // 24% RECALL on 5 query vectors
//        tmpDist = OC * norm(query) * tmpDist + QdC;   // ??% RECALL on 5 query vectors

        // ALT 2 (VoVAllen)
        // ⟨o𝑟, q𝑟⟩ = ⟨o, q⟩ · ∥o𝑟 − c∥ · ∥q𝑟 - c∥ + ⟨c, o𝑟⟩ + ⟨c, q𝑟 - c⟩
//         tmpDist = tmpDist * OC * QC + OdC + QmCdC; // 0% RECALL on 5 query vectors

        // ben
//         tmpDist = (qcDist * 2 - sumQ) * fac[facCounter].factorIP() * width; // regular rbq
//        tmpDist = (qcDist * 2) * fac[facCounter].factorIP() * width; // regular rbq w/o sumQ
//        tmpDist = distToC[bCounter] * y * tmpDist + QdC + OdC - Cnorm; // 0% REACLL on 5 query vectors
//        tmpDist = OdC * OC * tmpDist + QdC + OdC - Cnorm;

         // baseline
        float truth = VectorUtil.scaleMaxInnerProductScore(VectorUtil.dotProduct(O, query));   // 100% RECALL on 5 query vectors
//        tmpDist = truth;
        /////////////////////

        // FIXME: need more centroids
        // FIXME: validate the error bound
        float errorBound = y * fac[facCounter].error();
//        float errorBound = y * (fac[facCounter].error()) * 0.1f;
//        float errorBound = y * (fac[facCounter].error()) / (float) Math.pow(Cnorm, 2);
//        float estimator = tmpDist - errorBound;
        float estimator = tmpDist + errorBound;
//        float estimator = tmpDist; // - errorBound;
        estimators[i] = estimator;
//        float estimator = truth;  // 100% RECALL on 5 query vectors

        ////////////////////
        //FIXME: OPERATE ON ESTIMATOR INSTEAD OF TMPDIST??? ... invert here???
//        estimator = VectorUtil.scaleMaxInnerProductScore(estimator);
        ////////////////////

        if (estimatorDistances.size() < maxEstimatorSize) {
          totalEstimatorQueueAdds++;
          estimatorDistances.add(new Result(estimator, startC + i));
//        } else if (estimator < estimatorDistances.peek().sqrY()) {
        } else if (estimator > estimatorDistances.peek().sqrY()) {
          totalEstimatorQueueAdds++;
          estimatorDistances.poll();
          estimatorDistances.add(new Result(estimator, startC + i));
        }

        errorBoundAvg += errorBound;
        errorBoundTotalCalcs++;
        bCounter++;
        facCounter++;
      }
//      System.out.println("avg rbq diff: " + avgDiff / len[c]);
      try (BufferedWriter outputWriter = new BufferedWriter(new FileWriter("tmpdists.out"));) {
        outputWriter.write(Arrays.toString(tmpdists));
      }
      try (BufferedWriter outputWriter = new BufferedWriter(new FileWriter("rbqs.out"));) {
        outputWriter.write(Arrays.toString(rbqs));
      }
      try (BufferedWriter outputWriter = new BufferedWriter(new FileWriter("fests.out"));) {
        outputWriter.write(Arrays.toString(estimators));
      }
      try (BufferedWriter outputWriter = new BufferedWriter(new FileWriter("reals.out"));) {
        outputWriter.write(Arrays.toString(reals));
      }
      try (BufferedWriter outputWriter = new BufferedWriter(new FileWriter("rbqests.out"));) {
        outputWriter.write(Arrays.toString(rbqEsts));
      }
//      System.out.println("dists: " + tmpdists);
    }

    int size = estimatorDistances.size();
    for (int i = 0; i < size; i++) {
      Result res = estimatorDistances.remove();
//      if (res.sqrY() < distK) {
      if (res.sqrY() > distK) {
        floatingPointOps++;
        float gt_dist =
//            VectorUtils.squareDistance(dataVectors.vectorValue(dataMapping[res.c()]), query);
//            VectorUtil.dotProduct(dataVectors.vectorValue(dataMapping[res.c()]), query);
                mip(dataVectors.vectorValue(dataMapping[res.c()]), query);
//        if (gt_dist < distK) {
        if (gt_dist > distK) {
          knns.add(new Result(gt_dist, id[res.c()]));
          if (knns.size() > k) {
            knns.remove();
          }
          if (knns.size() == k) {
            distK = knns.peek().sqrY();
          }
        }
      }
    }

    IVFRNStats stats =
        new IVFRNStats(
            maxEstimatorSize,
            totalEstimatorQueueAdds,
            floatingPointOps,
            errorBoundAvg / errorBoundTotalCalcs);
    return new IVFRNResult(knns, stats);
  }

  public static float mip(float[] a, float[] b) {
    return VectorUtil.scaleMaxInnerProductScore(VectorUtil.dotProduct(a, b));
  }

  public static float[] divide(float[] a, float b) {
    float[] c = new float[a.length];
    for (int j = 0; j < a.length; j++) {
      c[j] = a[j] / b;
    }
    return c;
  }

  public static float[] subtract(float[] a, float[] b) {
    float[] c = new float[a.length];
    for (int j = 0; j < a.length; j++) {
      c[j] = a[j] - b[j];
    }
    return c;
  }

  public static float norm(float[] vector) {
    return MatrixUtils.partialNormForRow(vector);
  }

  public int getC() {
    return this.C;
  }

  public int getN() {
    return this.N;
  }
}
