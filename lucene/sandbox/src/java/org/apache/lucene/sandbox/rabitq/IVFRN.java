package org.apache.lucene.sandbox.rabitq;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.Random;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

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
  public long[][] binaryCode; // (B / 64) * N of 64-bit uint64_t

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
      long[][] binary,
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
    this.binaryCode = new long[N][B / 64];
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
      long[][] binaryCode,
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
        for (int j = 0; j < B / 64; j++) {
          bb.putLong(binaryCode[i][j]);
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

      long[][] binaryCode = new long[N][B / 64];
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
        for (int j = 0; j < B / 64; j++) {
          binaryCode[i][j] = bb.getLong();
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
        double x_x0 = distToC[i] / x0[i];
        float sqrX = distToC[i] * distToC[i];
        float error = (float) (2.0 * max_x1 * Math.sqrt(x_x0 * x_x0 - distToC[i] * distToC[i]));
        float factorPPC =
            (float)
                (-2.0
                    / fac_norm
                    * x_x0
                    * ((float) SpaceUtils.popcount(binaryCode[i], B) * 2.0 - B));
        float factorIP = (float) (-2.0 / fac_norm * x_x0);
        fac[i] = new Factor(sqrX, error, factorPPC, factorIP);
      }

      return new IVFRN(
          N, D, C, B, centroids, dataMapping, binaryCode, start, len, id, distToC, x0, u, fac);
    }
  }

  record QuantizedQuery(
      long[] result, int sumQ, float centroidDist, float vl, float width, int centroidId) {}

  public int getCentroidId(int vectorNodeId) {
    int centroidPos = Arrays.binarySearch(start, vectorNodeId);
    if (centroidPos == 0) {
      return 0;
    }
    assert centroidPos < 0;
    // Flip the sign and subtract 2 to get the centroid id
    return -centroidPos - 2;
  }

  public QuantizedQuery[] quantizeQuery(float[] query, int B_QUERY) {
    QuantizedQuery[] quantizedQueries = new QuantizedQuery[C];
    for (int c = 0; c < C; c++) {
      float sqrY = VectorUtils.squareDistance(query, centroids[c]);

      // Preprocess the residual query and the quantized query
      float[] v = SpaceUtils.range(query, centroids[c]);
      float vl = v[0], vr = v[1];
      float width = (vr - vl) / ((1 << B_QUERY) - 1);

      QuantResult quantResult = SpaceUtils.quantize(query, centroids[c], u, vl, width);
      byte[] byteQuery = quantResult.result();
      int sumQ = quantResult.sumQ();

      long[] quantQuery = SpaceUtils.transposeBin(byteQuery, D, B_QUERY);
      quantizedQueries[c] = new QuantizedQuery(quantQuery, sumQ, sqrY, vl, width, c);
    }
    return quantizedQueries;
  }

  public float quantizeCompare(QuantizedQuery quantizedQuery, int nodeId, int B_QUERY) {
    int c = quantizedQuery.centroidId();
    float sqrY = quantizedQuery.centroidDist();
    float vl = quantizedQuery.vl();
    float width = quantizedQuery.width();
    long[] quantQuery = quantizedQuery.result();
    int sumQ = quantizedQuery.sumQ();

    int startC = start[c];
    assert nodeId >= startC && nodeId < startC + len[c];

    float tmpDist = 0;
    long qcDist = SpaceUtils.ipByteBin(quantQuery, binaryCode[nodeId], B_QUERY, B);

    tmpDist +=
        fac[nodeId].sqrX()
            + sqrY
            + fac[nodeId].factorPPC() * vl
            + (qcDist * 2 - sumQ) * fac[nodeId].factorIP() * width;
    return tmpDist;
  }

  public IVFRNResult search(
      RandomAccessVectorValues.Floats dataVectors, float[] query, Arena offHeap, MemorySegment[] binaryCodeP, int k, int nProbe, int B_QUERY)
      throws IOException {
    // FIXME: FUTURE - implement fast scan and do a comparison

    assert nProbe < C;
    float distK = Float.MAX_VALUE;
    PriorityQueue<Result> knns = new PriorityQueue<>(k, Comparator.reverseOrder());

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
    PriorityQueue<Result> estimatorDistances =
            new PriorityQueue<>(maxEstimatorSize, Comparator.reverseOrder());

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
      float width = (vr - vl) / ((1 << B_QUERY) - 1);

      QuantResult quantResult = SpaceUtils.quantize(query, centroids[c], u, vl, width);
      byte[] byteQuery = quantResult.result();
      int sumQ = quantResult.sumQ();

      // Binary String Representation
      long[] quantQuery = SpaceUtils.transposeBin(byteQuery, D, B_QUERY);

      int startC = start[c];
      float y = (float) Math.sqrt(sqrY);

      int facCounter = startC;
      int bCounter = startC;

      MemorySegment quantQueryP = offHeap.allocateArray(ValueLayout.JAVA_LONG, quantQuery);

      for (int i = 0; i < len[c]; i++) {
        long qcDist = IpByteBinLibrary.ipByteBinNative(quantQueryP, binaryCodeP[bCounter], B);

        float tmpDist =
                fac[facCounter].sqrX()
                        + sqrY
                        + fac[facCounter].factorPPC() * vl
                        + (qcDist * 2 - sumQ) * fac[facCounter].factorIP() * width;
        float errorBound = y * (fac[facCounter].error());
        float estimator = tmpDist - errorBound;

        if (estimatorDistances.size() < maxEstimatorSize) {
          totalEstimatorQueueAdds++;
          estimatorDistances.add(new Result(estimator, startC + i));
        } else if (estimator < estimatorDistances.peek().sqrY()) {
          totalEstimatorQueueAdds++;
          estimatorDistances.poll();
          estimatorDistances.add(new Result(estimator, startC + i));
        }

        errorBoundAvg += errorBound;
        errorBoundTotalCalcs++;
        bCounter++;
        facCounter++;
      }
    }

    int size = estimatorDistances.size();
    for (int i = 0; i < size; i++) {
      Result res = estimatorDistances.remove();
      if (res.sqrY() < distK) {
        floatingPointOps++;
        float gt_dist =
                VectorUtils.squareDistance(dataVectors.vectorValue(dataMapping[res.c()]), query);
        if (gt_dist < distK) {
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

  public int getC() {
    return this.C;
  }
}
