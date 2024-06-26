package org.apache.lucene.sandbox.rabitq;

import org.apache.lucene.util.Constants;

// FIXME: FUTURE - copied from lucene internals for now
public class VectorUtils {

    public static float squareDistance(float[] a, float[] b) {
        float res = 0;
        int i = 0;

        // if the array is big, unroll it
        if (a.length > 32) {
            float acc1 = 0;
            float acc2 = 0;
            float acc3 = 0;
            float acc4 = 0;

            int upperBound = a.length & ~(4 - 1);
            for (; i < upperBound; i += 4) {
                // one
                float diff1 = a[i] - b[i];
                acc1 = fma(diff1, diff1, acc1);

                // two
                float diff2 = a[i + 1] - b[i + 1];
                acc2 = fma(diff2, diff2, acc2);

                // three
                float diff3 = a[i + 2] - b[i + 2];
                acc3 = fma(diff3, diff3, acc3);

                // four
                float diff4 = a[i + 3] - b[i + 3];
                acc4 = fma(diff4, diff4, acc4);
            }
            res += acc1 + acc2 + acc3 + acc4;
        }

        for (; i < a.length; i++) {
            float diff = a[i] - b[i];
            res = fma(diff, diff, res);
        }
        return res;
    }

    private static float fma(float a, float b, float c) {
        if (Constants.HAS_FAST_SCALAR_FMA) {
            return Math.fma(a, b, c);
        } else {
            return a * b + c;
        }
    }

}
