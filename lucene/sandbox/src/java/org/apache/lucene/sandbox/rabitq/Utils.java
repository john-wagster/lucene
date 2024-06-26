package org.apache.lucene.sandbox.rabitq;

import java.util.PriorityQueue;

public class Utils {
    public static double sqrtNewtonRaphson(double x, double curr, double prev) {
        return (curr == prev) ? curr : sqrtNewtonRaphson(x, 0.5 * (curr + x / curr), curr);
    }

    public static double constSqrt(double x) {
        return x >= 0 && !Double.isInfinite(x)
                ? sqrtNewtonRaphson(x, x, 0)
                : Double.NaN;
    }

    public static int reverseBits(int n) {
        n = (n >>> 1) & 0x55555555 | (n << 1) & 0xaaaaaaaa;
        n = (n >>> 2) & 0x33333333 | (n << 2) & 0xcccccccc;
        n = (n >>> 4) & 0x0f0f0f0f | (n << 4) & 0xf0f0f0f0;
        n = (n >>> 8) & 0x00ff00ff | (n << 8) & 0xff00ff00;
        n = (n >>> 16) & 0x0000ffff | (n << 16) & 0xffff0000;
        return n;
    }

    public static float getRatio(int q, float[][] Q, float[][] X, int[][] G, PriorityQueue<Result> KNNs) {
        PriorityQueue<Result> gt = new PriorityQueue<>();
        int k = KNNs.size();
        for (int i = 0; i < k; i++) {
            float sqrY = MatrixUtils.distance(Q, q, X, G[q][i]);
            int c = G[q][i];
            gt.add(new Result(sqrY, c));
        }
        double ret = 0;
        int validK = 0;
        while (!gt.isEmpty()) {
            if (gt.peek().sqrY() > 1e-5) {
                ret += Math.sqrt((float) KNNs.peek().sqrY() / gt.peek().sqrY());
                validK++;
            }
            gt.remove();
            KNNs.remove();
        }
        if (validK == 0) return k;
        return (float) (ret / validK * k);
    }


}
