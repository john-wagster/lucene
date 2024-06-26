package org.apache.lucene.sandbox.rabitq;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.PriorityQueue;


public class IVFRN {
    public Factor[] fac;

    public int N;                        // the number of data vectors
    public int C;                        // the number of clusters

    public int[] start;                  // the start point of a cluster
    public int[] len;                    // the length of a cluster
    public int[] id;                     // N of size_t the ids of the objects in a cluster
    public float[] distToC;              // N of floats distance to the centroids (not the squared distance)
    public float[] u;                    // B of floats random numbers sampled from the uniform distribution [0,1]

    // FIXME: FUTURE - make this a byte[] instead??
    public long[][] binaryCode;          // (B / 64) * N of 64-bit uint64_t

    public float[] x0;                   // N of floats in the Random Net algorithm
    public float[][] centroids;          // N * B floats (not N * D), note that the centroids should be randomized
    public float[][] data;               // N * D floats, note that the datas are not randomized

    private static int B;
    private static int D;
    private static float fac_norm = (float) Utils.constSqrt(1.0 * B);
    private static float max_x1 = (float) (1.9 / Utils.constSqrt(1.0 * B-1.0));

    public IVFRN(float[][] X, float[][] centroids, float[] distToCentroid, float[] _x0, int[] clusterId, long[][] binary) {

        // FIXME: FUTURE - compute fac and u here??

        D = X[0].length;
        B = (D + 63) / 64 * 64;

        N = X.length;
        C = centroids.length;

        // Check if B is a multiple of 64
        assert (B % 64 == 0);

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
        data = X;
        binaryCode = binary;
    }

    private IVFRN(int n, int d, int c, int b, float[][] centroids, float[][] data, long[][] binaryCode, int[] start,
                  int[] len, int[] id, float[] distToC, float[] x0, float[] u, Factor[] fac) {
        this.N = n;
        this.D = d;
        this.C = c;
        this.B = b;
        this.centroids = centroids;
        this.data = data;
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

        try(FileOutputStream fos = new FileOutputStream(filename);
            DataOutputStream dos = new DataOutputStream(fos)) {

            dos.writeInt(N);
            dos.writeInt(D);
            dos.writeInt(C);
            dos.writeInt(B);

            for (int i = 0; i < N; i++) {
                dos.writeInt(start[i]);
            }

            for (int i = 0; i < N; i++) {
                dos.writeInt(len[i]);
            }

            for (int i = 0; i < N; i++) {
                dos.writeInt(id[i]);
            }

            for (int i = 0; i < N; i++) {
                dos.writeFloat(distToC[i]);
            }

            for (int i = 0; i < N; i++) {
                dos.writeFloat(x0[i]);
            }

            for (int i = 0; i < C; i++) {
                for (int j = 0; j < B; j++) {
                    dos.writeFloat(centroids[i][j]);
                }
            }

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < D; j++) {
                    dos.writeFloat(data[i][j]);
                }
            }

            for (int i = 0; i < B / 64; i++) {
                for (int j = 0; j < N; j++) {
                    dos.writeLong(binaryCode[i][j]);
                }
            }
        }
    }

    public static IVFRN load(String filename) throws IOException {
        try (FileInputStream fis = new FileInputStream(filename);
             DataInputStream dis = new DataInputStream(fis) ) {

            int N = dis.readInt();
            int D = dis.readInt();
            int C = dis.readInt();
            int B = dis.readInt();

            float[][] centroids = new float[C][B];
            float[][] data = new float[N][D];

            long[][] binaryCode = new long[B / 64][N];
            int[] start = new int[C];
            int[] len = new int[C];
            int[] id = new int[N];
            float[] distToC = new float[N];
            float[] x0 = new float[N];

            for (int i = 0; i < N; i++) {
                start[i] = dis.readInt();
            }

            for (int i = 0; i < N; i++) {
                len[i] = dis.readInt();
            }

            for (int i = 0; i < N; i++) {
                id[i] = dis.readInt();
            }

            for (int i = 0; i < N; i++) {
                distToC[i] = dis.readFloat();
            }

            for (int i = 0; i < N; i++) {
                x0[i] = dis.readFloat();
            }

            for (int i = 0; i < C; i++) {
                for (int j = 0; j < B; j++) {
                    centroids[i][j] = dis.readFloat();
                }
            }

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < D; j++) {
                    data[i][j] = dis.readFloat();
                }
            }

            for (int i = 0; i < B / 64; i++) {
                for (int j = 0; j < N; j++) {
                    binaryCode[i][j] = dis.readLong();
                }
            }

            float[] u = new float[B];
            for (int i = 0; i < B; i++) {
                u[i] = (float) Math.random();
            }

            Factor[] fac = new Factor[N];
            for (int i = 0; i < N; i++) {
                double x_x0 = distToC[i] / x0[i];
                float sqrX = distToC[i] * distToC[i];
                float error = (float) (2.0 * max_x1 * Math.sqrt(x_x0 * x_x0 - distToC[i] * distToC[i]));
                float factorPPC = (float) (-2.0 / fac_norm * x_x0 * ((float) SpaceUtils.popcount(binaryCode[i]) * 2.0 - B));
                float factorIP = (float) (-2.0 / fac_norm * x_x0);
                fac[i] = new Factor(sqrX, error, factorPPC, factorIP);
            }

            return new IVFRN(N, D, C, B, centroids, data, binaryCode, start, len, id, distToC, x0, u, fac);
        }
    }

    public PriorityQueue<Result> search(float[] query, float[] rdQuery, int k, int nProbe, float distK, int B_QUERY) {
        //FIXME: FUTURE - implement fast scan and do a comparison

        PriorityQueue<Result> knns = new PriorityQueue<>();

        // Find out the nearest N_{probe} centroids to the query vector.
        Result[] centroidDist = new Result[C];
        for (int i = 0; i < C; i++) {
            centroidDist[i] = new Result(VectorUtils.squareDistance(rdQuery, centroids[i]), i);
        }

        Arrays.sort(centroidDist, 0, nProbe);

        for (int pb = 0; pb < nProbe; pb++) {
            int c = centroidDist[pb].c();
            float sqrY = centroidDist[pb].sqrY();

            // Preprocess the residual query and the quantized query
            float[] v = SpaceUtils.range(rdQuery, centroids[c]);
            float vl = v[0], vr = v[1];
            float width = (vr - vl) / ((1 << B_QUERY) - 1);

            QuantResult quantResult = SpaceUtils.quantize(rdQuery, centroids[c], u, vl, width);
            byte[] byteQuery = quantResult.result();
            int sumQ = quantResult.sumQ();

            // Binary String Representation
            long[] quantQuery = SpaceUtils.transposeBin(byteQuery, D, B_QUERY);

            int startC = start[c];
            scan(knns, distK, k,
                    quantQuery, startC, len[c],
                    sqrY, vl, width, sumQ,
                    query, B_QUERY, B);
        }

        return knns;
    }

    public void scan(PriorityQueue<Result> KNNs, float distK, int k, long[] quantQuery, int startC, int len,
                     float sqr_y, float vl, float width, float sumq, float[] query, int B_QUERY, int B) {
        int SIZE = 32;
        float y = (float) Math.sqrt(sqr_y);
        float[] res = new float[SIZE];
        int it = len / SIZE;

        int nextC = startC;
        int dataC = startC;
        int idC = startC;

        // FIXME: FUTURE - had to flatten this to get it working
        long[] flattendedBinaryCode = MatrixUtils.flatten(binaryCode);

        int nextBinCodeStart = 0;
        for (int i = 0; i < it; i++) {
            for (int j = 0; j < SIZE; j++) {
                // FIXME: FUTURE - clean this up -- this is unnecessary
                long[] subFlatBinCodes = Arrays.copyOfRange(flattendedBinaryCode, nextBinCodeStart, j * B/64+B/64);
                float tmp_dist = (fac[nextC].sqrX()) + sqr_y + fac[nextC].factorPPC() * vl +
                        (SpaceUtils.ipByteBin(quantQuery, subFlatBinCodes, B_QUERY, B) * 2 - sumq) *
                                (fac[nextC].factorIP()) * width;
                nextBinCodeStart += B / 64;
                float error_bound = y * (fac[nextC].error());
                res[j] = tmp_dist - error_bound;
                nextC++;
            }

            for (int j = 0; j < SIZE; j++) {
                if (res[j] < distK) {
                    float gt_dist = VectorUtils.squareDistance(query, data[dataC]);
                    if (gt_dist < distK) {
                        KNNs.add(new Result(gt_dist, id[idC]));
                        if (KNNs.size() > k) KNNs.remove();
                        if (KNNs.size() == k) distK = KNNs.peek().sqrY();
                    }
                }
                dataC++;
                idC++;
            }
        }

        // FIXME: FUTURE - had to flatten this to get it working
        flattendedBinaryCode = MatrixUtils.flatten(binaryCode);

        nextBinCodeStart = 0;
        for (int i = it * SIZE, j=0; i < len; i++, j++) {
            long[] subFlatBinCodes = Arrays.copyOfRange(flattendedBinaryCode, nextBinCodeStart, j * B/64+B/64);
            float tmpDist = (fac[nextC].sqrX()) + sqr_y + fac[nextC].factorPPC() * vl +
                    (SpaceUtils.ipByteBin(quantQuery, subFlatBinCodes, B_QUERY, B) * 2 - sumq) * (fac[nextC].factorIP()) * width;
            float errorBound = y * (fac[nextC].error());
            res[j] = tmpDist - errorBound;
            nextC++;
            nextBinCodeStart += B / 64;
        }

        for (int i = it * SIZE, j=0; i < len; i++, j++) {
            if (res[j] < distK) {
                float gt_dist = VectorUtils.squareDistance(query, data[dataC]);
                if (gt_dist < distK) {
                    KNNs.add(new Result(gt_dist, id[idC]));
                    if (KNNs.size() > k) KNNs.remove();
                    if (KNNs.size() == k) distK = KNNs.peek().sqrY();
                }
            }
            dataC++;
            idC++;
        }
    }
}

