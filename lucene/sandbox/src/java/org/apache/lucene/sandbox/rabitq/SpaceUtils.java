package org.apache.lucene.sandbox.rabitq;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;

public class SpaceUtils {

    public static int popcount(long[] d) {
        int ret = 0;
        for (int i = 0; i < d.length / 8; i++) {
            ret += Long.bitCount(d[i]);
        }
        return ret;
    }

    public static long ipByteBin(long[] q, long[] d, int B_QUERY, int B) {
        long ret = 0;
        for (int i = 0; i < B_QUERY; i++) {
            long subRet = 0;
            for (int j = 0; j < B / 64; j++) {
                subRet += Long.bitCount(q[i*(B / 64)+j] & d[j]);
            }
            ret += subRet << i;
        }
        return ret;
    }

    public static long[] transposeBin(byte[] q, int D, int B_QUERY){
        int B = (D + 63) / 64 * 64;

        long[] quantQuery = new long[B_QUERY * B / 64];

        for(int i=0;i<B;i+=32){
            ByteBuffer buffer = ByteBuffer.wrap(q, i, 32);
            IntBuffer intBuffer = buffer.asIntBuffer(); // Interpret the bytes as int values

            int[] v = new int[8];
            for (int j = 0; j < 8; j++) {
                v[j] = Integer.rotateLeft(intBuffer.get(), 8 - B_QUERY); // Load and shift the int value
            }

            for(int j=0;j<B_QUERY;j++){
                long[] v1 = new long[8];
                for (int k = 0; k < 8; k++) {
                    v1[k] = reverseBits(v[k]); // Reverse the bits of each int value
                    v[k] <<= 8; // Shift left by 8 bits for the next iteration
                }

                long mask = 0L;
                for (int k = 7; k >= 0; k--) {
                    mask |= v1[k]; // Combine the reversed int values into a single uint64_t value
                    if (k > 0) {
                        mask <<= 8;
                    }
                }

                quantQuery[(B_QUERY - j - 1) * (B / 64) + i / 64] |= (mask << ((i / 32 % 2 == 0) ? 32 : 0)); // Apply bitwise OR and shifting operations
            }
        }

        return quantQuery;
    }

    private static long reverseBits(int n) {
        n = (n >>> 1) & 0x55555555 | (n << 1) & 0xaaaaaaaa;
        n = (n >>> 2) & 0x33333333 | (n << 2) & 0xcccccccc;
        n = (n >>> 4) & 0x0f0f0f0f | (n << 4) & 0xf0f0f0f0;
        n = (n >>> 8) & 0x00ff00ff | (n << 8) & 0xff00ff00;
        n = (n >>> 16) & 0x0000ffff | (n << 16) & 0xffff0000;
        return Integer.toUnsignedLong(n);
    }

    public static void main(String[] args) {
        long v = reverseBits((int) reverseBits(Integer.MIN_VALUE));
        System.out.println(Integer.MIN_VALUE);
        System.out.println(Integer.MAX_VALUE);
        System.out.println(v);
    }

    public static float[] range(float[] q, float[] c) {
        float vl = Float.POSITIVE_INFINITY;
        float vr = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < q.length; i++) {
            float tmp = q[i] - c[i];
            if (tmp < vl) {
                vl = tmp;
            }
            if (tmp > vr) {
                vr = tmp;
            }
        }

        return new float[] {vl, vr};
    }

    public static QuantResult quantize(float[] q, float[] c, float[] u, float vl, float width) {
        byte[] result = new byte[q.length];
        float oneOverWidth = 1.0f / width;
        int sumQ = 0;
        for (int i = 0; i < q.length; i++) {
            byte res = (byte)(((q[i] - c[i]) - vl) * oneOverWidth + u[i]);
            result[i] = res;
            sumQ += res;
        }

        return new QuantResult(result, sumQ);
    }
}
