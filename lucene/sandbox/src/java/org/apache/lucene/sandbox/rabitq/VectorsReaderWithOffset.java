package org.apache.lucene.sandbox.rabitq;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

import java.io.IOException;

public class VectorsReaderWithOffset implements RandomAccessVectorValues.Floats {
  private final IndexInput slice;
  private final int size;
  private final int dim;
  private final int byteSize;
  private final int offset;
  private int lastOrd = -1;
  private final float[] value;

  public VectorsReaderWithOffset(IndexInput slice, int size, int dim, int byteSize, int offset) {
    this.slice = slice;
    this.size = size;
    this.dim = dim;
    this.byteSize = byteSize;
    this.offset = offset;
    value = new float[dim];
  }

  @Override
  public int dimension() {
    return dim;
  }

  @Override
  public IndexInput getSlice() {
    return slice;
  }

  @Override
  public int ordToDoc(int ord) {
    throw new IllegalStateException("Not supported");
  }

  @Override
  public Bits getAcceptOrds(Bits acceptDocs) {
    throw new IllegalStateException("Not supported");
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public int getVectorByteLength() {
    return byteSize;
  }

  @Override
  public float[] vectorValue(int targetOrd) throws IOException {
    if (lastOrd == targetOrd) {
      return value;
    }
    long seekPos = offset + (long) targetOrd * byteSize;
    slice.seek(seekPos);
    slice.readFloats(value, 0, value.length);
    lastOrd = targetOrd;
    return value;
  }

  @Override
  public Floats copy() throws IOException {
    return new VectorsReaderWithOffset(slice, size, dim, byteSize, offset);
  }
}
