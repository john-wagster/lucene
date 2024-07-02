package org.apache.lucene.sandbox.rabitq;

import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.Arrays;

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

    public static void main(String[] args) {

        System.out.println((byte)(26 << 4));

        byte[] q = { 0x1a, (byte) 0x2b, (byte) 0x3c, (byte) 0x4d, (byte) 0x5e, (byte) 0x6f, (byte) 0x7a, (byte) 0x8b,
                (byte) 0x9c, (byte) 0xad, (byte) 0xbe, (byte) 0xcf, (byte) 0xda, (byte) 0xeb, (byte) 0xfc, (byte) 0x0d,
                (byte) 0x1e, (byte) 0x2f, (byte) 0x3a, (byte) 0x4b, (byte) 0x5c, (byte) 0x6d, (byte) 0x7e, (byte) 0x8f,
                (byte) 0x90, (byte) 0xa1, (byte) 0xb2, (byte) 0xc3, (byte) 0xd4, (byte) 0xe5, (byte) 0xf6, (byte) 0x07,
                (byte) 0x18, (byte) 0x29, (byte) 0x3a, (byte) 0x4b, (byte) 0x5c, (byte) 0x6d, (byte) 0x7e, (byte) 0x8f,
                (byte) 0x90, (byte) 0xa1, (byte) 0xb2, (byte) 0xc3, (byte) 0xd4, (byte) 0xe5, (byte) 0xf6, (byte) 0x07,
                (byte) 0x18, (byte) 0x29, (byte) 0x3a, (byte) 0x4b, (byte) 0x5c, (byte) 0x6d, (byte) 0x7e, (byte) 0x8f,
                (byte) 0x90, (byte) 0xa1, (byte) 0xb2, (byte) 0xc3, (byte) 0xd4, (byte) 0xe5, (byte) 0xf6, (byte) 0x07,
                (byte) 0x18, (byte) 0x29, (byte) 0x3a, (byte) 0x4b, (byte) 0x5c, (byte) 0x6d, (byte) 0x7e, (byte) 0x8f,
                (byte) 0x90, (byte) 0xa1, (byte) 0xb2, (byte) 0xc3, (byte) 0xd4, (byte) 0xe5, (byte) 0xf6, (byte) 0x07,
                (byte) 0x18, (byte) 0x29, (byte) 0x3a, (byte) 0x4b, (byte) 0x5c, (byte) 0x6d, (byte) 0x7e, (byte) 0x8f,
                (byte) 0x90, (byte) 0xa1, (byte) 0xb2, (byte) 0xc3, (byte) 0xd4, (byte) 0xe5, (byte) 0xf6, (byte) 0x07,
                (byte) 0x18, (byte) 0x29, (byte) 0x3a, (byte) 0x4b, (byte) 0x5c, (byte) 0x6d, (byte) 0x7e, (byte) 0x8f,
                (byte) 0x90, (byte) 0xa1, (byte) 0xb2, (byte) 0xc3, (byte) 0xd4, (byte) 0xe5, (byte) 0xf6, (byte) 0x07,
                (byte) 0x18, (byte) 0x29, (byte) 0x3a, (byte) 0x4b, (byte) 0x5c, (byte) 0x6d, (byte) 0x7e, (byte) 0x8f,
                (byte) 0x90, (byte) 0xa1, (byte) 0xb2, (byte) 0xc3, (byte) 0xd4, (byte) 0xe5, (byte) 0xf6, (byte) 0x07 };
//        long[] tq = new long[q.length / 8]; // Output buffer
//        Arrays.fill(tq, 0);
        long[] tq = transposeBin(q, 128, 4);
        for(int i = 0; i < tq.length; i++) {
            System.out.print(Long.toUnsignedString(tq[i]) + " ");
        }
        // 6148914691236517205 6148914691236517205 14933077865681597235 3689348814741910323 4392081725200469775 1085102592571150095 18446742978476179200 18374966859414961920
    }

    public static long[] transposeBin(byte[] q, int D, int B_QUERY) {
        assert B_QUERY == 4;  // see FIXME below

        int B = (D + 63) / 64 * 64;
        long[] quantQuery = new long[B_QUERY * B / 64];


        int qOffset = 0;
        for(int i = 0; i < B; i+=32) {

            byte[] v = new byte[32];

            // for every four bytes we shift left (with remainder across those bytes)
            int shift = 8 - B_QUERY;
            for (int j = 0; j < v.length; j += 4) {
                byte[] s = new byte[4];
                // FIXME: these masks only work for B_QUERY = 4 ... fix hardcoded masking
                s[0] = (byte) (q[qOffset + j] << shift);
                s[1] = (byte) (q[qOffset + j + 1] << shift | ((q[qOffset + j] >>> (8-shift))  & 0b00001111));
                s[2] = (byte) (q[qOffset + j + 2] << shift | ((q[qOffset + j + 1] >>> (8-shift) & 0b00001111)));
                s[3] = (byte) (q[qOffset + j + 3] << shift | ((q[qOffset + j + 2] >>> (8-shift) & 0b00001111)));

                v[j] = s[0];
                v[j + 1] = s[1];
                v[j + 2] = s[2];
                v[j + 3] = s[3];
            }

            for (int j = 0; j < B_QUERY; j++) {
                long v1 = moveMaskEpi8(v);
                // v1 = reverseBits(v1); // our move mask does this operation for us
                quantQuery[(B_QUERY - j - 1) * (B / 64) + i / 64] |= (v1 << ((i / 32 % 2 == 0) ? 32:0));

                for(int k = 0; k < v.length; k+=4) {
                    ByteBuffer bb = ByteBuffer.allocate(4);
                    for(int l = 3; l >= 0; l--) {
                        bb.put(v[k+l]);
                    }
                    bb.flip();
                    int value = bb.getInt();
                    value += value;
                    byte[] sumSubV = ByteBuffer.allocate(4).putInt(value).array();
                    v[k] = sumSubV[3];
                    v[k+1] = sumSubV[2];
                    v[k+2] = sumSubV[1];
                    v[k+3] = sumSubV[0];
                }
            }
            qOffset += 32;
        }

        return quantQuery;
    }

    private static long moveMaskEpi8(byte[] v) {
        long v1 = 0;
        for(int k = 0; k < v.length; k++) {
            if((v[k] & 0b10000000) == 0b10000000) {
                v1 |= 0b00000001;
            } else {
                v1 |= 0b00000000;
            }
            if(k != v.length-1) {
                v1 <<= 1;
            }
        }

        return v1;
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
