package org.apache.lucene.sandbox.rabitq;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Map;

public class SpaceUtils {

  private static final VectorSpecies<Byte> BYTE_SPECIES = ByteVector.SPECIES_PREFERRED;

  public static int popcount(byte[] d, int B) {
    return BitSet.valueOf(d).cardinality();
//    for (int i = 0; i < B / 8; i++) {
//      // FIXME: implement this in C because Long.bitCount is not optimized
//      // FIXME: Integer.bitCount used like this is still slower than Long.bitCount
//      // FIXME: a more comprehensive refactor to remove longs and replace with ints is non-trivial
//      ret += BitSet.valueOf(d).cardinality();
////      ret += Long.bitCount(d[i]);
//      //            ret += Integer.bitCount((int)(d[i] & 0x00000000ffffffffL));
//      //            ret += Integer.bitCount((int)((d[i] & 0xffffffff00000000L) >>> 32));
//    }
//    return ret;
  }

  public static long ipByteBin(long[] q, long[] d, int B_QUERY, int B) {
    long ret = 0;
    int size = B / 64;
    for (int i = 0; i < B_QUERY; i++) {
      long subRet = 0;
      for (int j = 0; j < size; j++) {
        // FIXME: implement this in C because Long.bitCount is not optimized
        // FIXME: Integer.bitCount used like this is still slower than Long.bitCount
        // FIXME: a more comprehensive refactor to remove longs and replace with ints is non-trivial
        long estimatedDist = q[i * size + j] & d[j];
        subRet += Long.bitCount(estimatedDist);
        //                subRet += Integer.bitCount((int)(estimatedDist & 0x00000000ffffffffL));
        //                subRet += Integer.bitCount((int)((estimatedDist & 0xffffffff00000000L) >>>
        // 32));
      }
      ret += subRet << i;
    }
    return ret;
  }

  public static int countBits(byte b) {
    int count = 0;
    for (int i = 7; i >= 0; i--) {
      if (((b >> i) & 1) == 1) {
        count++;
      }
    }
    return count;
  }

  public static long ipByteBinByte(byte[] q, byte[] d, int B_QUERY, int B, Map<Byte, Integer> byteMap) {
    long ret = 0;
    int size = B / 8;
    for (int i = 0; i < B_QUERY; i++) {
      long subRet = 0;
      for (int j = 0; j < size; j++) {
        byte estimatedDist = (byte) (q[i * size + j] & d[j]);
        subRet += byteMap.get(estimatedDist);
      }
      ret += subRet << i;
    }
    return ret;
  }

  public static long ipByteBinBytePan(byte[] q, byte[] d, int B_QUERY, int B, Map<Byte, Integer> byteMap) {
    int vectorSize = d.length / BYTE_SPECIES.length();
    long ret = 0;
    for (int i = 0; i < B_QUERY; i++) {
      long subRet = 0;
      for (int r = 0; r < vectorSize; r++) {
        int offset = BYTE_SPECIES.length() * r;
        ByteVector vq = ByteVector.fromArray(BYTE_SPECIES, q, d.length * i + offset);
        ByteVector vd = ByteVector.fromArray(BYTE_SPECIES, d, offset);
        ByteVector vres = vq.and(vd);
        vres = vres.lanewise(VectorOperators.BIT_COUNT);
        subRet += vres.reduceLanes(VectorOperators.ADD); // subRet += byteMap.get(estimatedDist)
      }
      ret += subRet << i;
    }
    return ret;
  }

  public static byte[] transposeBin(byte[] q, int D, int B_QUERY) {
    // FIXME: WAGS REWRITE THIS FUNC
    // FIXME: FUTURE - verify B_QUERY > 0
    // FIXME: rewrite with panama?
    assert B_QUERY > 0;

    int B = (D + 63) / 64 * 64;
    long[] quantQuery = new long[B_QUERY * B / 64];

    int byte_mask = 1;
    for (int i = 0; i < B_QUERY - 1; i++) {
      byte_mask = byte_mask << 1 | 0b00000001;
    }

    int qOffset = 0;
    for (int i = 0; i < B; i += 32) {

      byte[] v = new byte[32];

      // for every four bytes we shift left (with remainder across those bytes)
      int shift = 8 - B_QUERY;
      for (int j = 0; j < v.length; j += 4) {
        byte[] s = new byte[4];
        s[0] = (byte) (q[qOffset + j] << shift);
        s[1] =
            (byte) (q[qOffset + j + 1] << shift | ((q[qOffset + j] >>> (8 - shift)) & byte_mask));
        s[2] =
            (byte)
                (q[qOffset + j + 2] << shift | ((q[qOffset + j + 1] >>> (8 - shift)) & byte_mask));
        s[3] =
            (byte)
                (q[qOffset + j + 3] << shift | ((q[qOffset + j + 2] >>> (8 - shift)) & byte_mask));

        v[j] = s[0];
        v[j + 1] = s[1];
        v[j + 2] = s[2];
        v[j + 3] = s[3];
      }

      for (int j = 0; j < B_QUERY; j++) {
        long v1 = moveMaskEpi8(v);
        // v1 = reverseBits(v1); // our move mask does this operation for us
        quantQuery[(B_QUERY - j - 1) * (B / 64) + i / 64] |= (v1 << ((i / 32 % 2 == 0) ? 32 : 0));

        for (int k = 0; k < v.length; k += 4) {
          ByteBuffer bb = ByteBuffer.allocate(4);
          for (int l = 3; l >= 0; l--) {
            bb.put(v[k + l]);
          }
          bb.flip();
          int value = bb.getInt();
          value += value;
          byte[] sumSubV = ByteBuffer.allocate(4).putInt(value).array();
          v[k] = sumSubV[3];
          v[k + 1] = sumSubV[2];
          v[k + 2] = sumSubV[1];
          v[k + 3] = sumSubV[0];
        }
      }
      qOffset += 32;
    }

    ByteBuffer bb = ByteBuffer.allocate((B / 8) * B_QUERY);
    for(int j = 0; j < quantQuery.length; j++) {
      bb.putLong(quantQuery[j]);
    }
    bb.flip();
    return bb.array();
  }

  private static long moveMaskEpi8(byte[] v) {
    long v1 = 0;
    for (int k = 0; k < v.length; k++) {
      if ((v[k] & 0b10000000) == 0b10000000) {
        v1 |= 0b00000001;
      } else {
        v1 |= 0b00000000;
      }
      if (k != v.length - 1) {
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
    // FIXME: speed up with panama?
    byte[] result = new byte[q.length];
    float oneOverWidth = 1.0f / width;
    int sumQ = 0;
    for (int i = 0; i < q.length; i++) {
      byte res = (byte) (((q[i] - c[i]) - vl) * oneOverWidth + u[i]);
      result[i] = res;
      sumQ += res;
    }

    return new QuantResult(result, sumQ);
  }
}
