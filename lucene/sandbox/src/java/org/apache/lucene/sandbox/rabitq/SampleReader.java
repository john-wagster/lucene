package org.apache.lucene.sandbox.rabitq;

import java.io.IOException;
import java.util.Random;
import java.util.function.IntUnaryOperator;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

/** A reader of vector values that samples a subset of the vectors. */
public class SampleReader implements RandomAccessVectorValues.Floats {
  private final RandomAccessVectorValues.Floats origin;
  private final int sampleSize;
  private final IntUnaryOperator sampleFunction;

  SampleReader(
      RandomAccessVectorValues.Floats origin, int sampleSize, IntUnaryOperator sampleFunction) {
    this.origin = origin;
    this.sampleSize = sampleSize;
    this.sampleFunction = sampleFunction;
  }

  @Override
  public int size() {
    return sampleSize;
  }

  @Override
  public int dimension() {
    return origin.dimension();
  }

  @Override
  public Floats copy() throws IOException {
    throw new IllegalStateException("Not supported");
  }

  @Override
  public IndexInput getSlice() {
    return origin.getSlice();
  }

  @Override
  public float[] vectorValue(int targetOrd) throws IOException {
    return origin.vectorValue(sampleFunction.applyAsInt(targetOrd));
  }

  @Override
  public int getVectorByteLength() {
    return origin.getVectorByteLength();
  }

  @Override
  public int ordToDoc(int ord) {
    throw new IllegalStateException("Not supported");
  }

  @Override
  public Bits getAcceptOrds(Bits acceptDocs) {
    throw new IllegalStateException("Not supported");
  }

  public static SampleReader createSampleReader(
      RandomAccessVectorValues.Floats origin, int k, long seed) {
    int[] samples = reservoirSample(origin.size(), k, seed);
    return new SampleReader(origin, samples.length, i -> samples[i]);
  }

  /**
   * Sample k elements from n elements according to reservoir sampling algorithm.
   *
   * @param n number of elements
   * @param k number of samples
   * @param seed random seed
   * @return array of k samples
   */
  public static int[] reservoirSample(int n, int k, long seed) {
    Random rnd = new Random(seed);
    int[] reservoir = new int[k];
    for (int i = 0; i < k; i++) {
      reservoir[i] = i;
    }
    for (int i = k; i < n; i++) {
      int j = rnd.nextInt(i + 1);
      if (j < k) {
        reservoir[j] = i;
      }
    }
    return reservoir;
  }

  /**
   * Sample k elements from the origin array using reservoir sampling algorithm.
   *
   * @param origin original array
   * @param k number of samples
   * @param seed random seed
   * @return array of k samples
   */
  public static int[] reservoirSampleFromArray(int[] origin, int k, long seed) {
    Random rnd = new Random(seed);
    if (k >= origin.length) {
      return origin;
    }
    int[] reservoir = new int[k];
    for (int i = 0; i < k; i++) {
      reservoir[i] = origin[i];
    }
    for (int i = k; i < origin.length; i++) {
      int j = rnd.nextInt(i + 1);
      if (j < k) {
        reservoir[j] = origin[i];
      }
    }
    return reservoir;
  }
}
