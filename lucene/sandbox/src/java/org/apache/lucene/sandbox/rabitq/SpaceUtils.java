package org.apache.lucene.sandbox.rabitq;

import java.nio.ByteBuffer;
import java.util.BitSet;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.apache.lucene.util.BitUtil;

public class SpaceUtils {

  private static final VectorSpecies<Byte> BYTE_SPECIES = ByteVector.SPECIES_PREFERRED;

  // 𝐵𝑞 = Θ(log log𝐷)
  public static int B_QUERY = 4;

  public static int popcount(byte[] d, int B) {
    return BitSet.valueOf(d).cardinality();
  }

  public static long ipByteBinByte(byte[] q, byte[] d) {
    long ret = 0;
    int size = d.length;
    for (int i = 0; i < B_QUERY; i++) {
      int r = 0;
      long subRet = 0;
      for (final int upperBound = d.length & -Integer.BYTES; r < upperBound; r += Integer.BYTES) {
        subRet +=
            Integer.bitCount(
                (int) BitUtil.VH_NATIVE_INT.get(q, i * size + r)
                    & (int) BitUtil.VH_NATIVE_INT.get(d, r));
      }
      for (; r < d.length; r++) {
        subRet += Integer.bitCount((q[i * size + r] & d[i]) & 0xFF);
      }
      ret += subRet << i;
    }
    return ret;
  }

  public static long ipByteBinBytePan(byte[] q, byte[] d) {
    int vectorSize = d.length / BYTE_SPECIES.length();
    long ret = 0;
    int size = d.length;
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

      // FIXME: come back and pad the arrays with zeros instead of dealing with the tail?
      // tail
      int remainder = d.length % BYTE_SPECIES.length();
      if(remainder != 0) {
        for(int j = d.length-remainder; j < d.length; j += Integer.BYTES) {
          subRet += Integer.bitCount((int) BitUtil.VH_NATIVE_INT.get(q, i * size + j) & (int) BitUtil.VH_NATIVE_INT.get(d, j));
        }
      }

      ret += subRet << i;
    }


    return ret;
  }

  public static byte[] transposeBinByte(byte[] q, int D) {
    // FIXME: rewrite this function to no longer use longs
    // FIXME: FUTURE - verify B_QUERY > 0
    // FIXME: rewrite with panama?
    assert B_QUERY > 0;

    int B = (D + 63) / 64 * 64;
    byte[] quantQueryByte = new byte[B_QUERY * B / 8];

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
        s[0] = (byte) (q[qOffset + j] << shift | ((q[qOffset + j] >>> (8 - shift)) & byte_mask));
        s[1] = (byte) (q[qOffset + j + 1] << shift | ((q[qOffset + j + 1] >>> (8 - shift)) & byte_mask));
        s[2] = (byte) (q[qOffset + j + 2] << shift | ((q[qOffset + j + 2] >>> (8 - shift)) & byte_mask));
        s[3] = (byte) (q[qOffset + j + 3] << shift | ((q[qOffset + j + 3] >>> (8 - shift)) & byte_mask));

        v[j] = s[0];
        v[j + 1] = s[1];
        v[j + 2] = s[2];
        v[j + 3] = s[3];
      }

      for (int j = 0; j < B_QUERY; j++) {
        byte[] v1 = moveMaskEpi8Byte(v);
        for (int k = 0; k < 4; k++) {
          quantQueryByte[(B_QUERY - j - 1) * (B / 8) + i / 8 + k] = v1[k];
        }

        for (int k = 0; k < v.length; k += 4) {
          v[k] = (byte) (v[k] + v[k]);
          v[k+1] = (byte) (v[k+1] + v[k+1]);
          v[k+2] = (byte) (v[k+2] + v[k+2]);
          v[k+3] = (byte) (v[k+3] + v[k+3]);
        }
      }
      qOffset += 32;
    }

    return quantQueryByte;
  }

  private static byte[] moveMaskEpi8Byte(byte[] v) {
    byte[] v1b = new byte[4];
    int m = 0;
    for (int k = 0; k < v.length; k++) {
      if ((v[k] & 0b10000000) == 0b10000000) {
        v1b[m] |= 0b00000001;
      }
      if (k % 8 == 7) {
        m++;
      } else {
        v1b[m] <<= 1;
      }
    }

    return v1b;
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
