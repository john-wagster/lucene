package org.apache.lucene.sandbox.rabitq;

import java.io.IOException;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;

public class OffHeapRBQRandomVectorScorerSupplier implements RandomVectorScorerSupplier {

  private final RandomAccessVectorValues.Floats rawVectorValues;
  // Right now, these are on heap
  private final IVFRN quantizedVectorValues;
  final long quantizedByteSize;
  final int quantizedDimSize;
  final IndexInput slice;
  final int bquery;

  public OffHeapRBQRandomVectorScorerSupplier(
      RandomAccessVectorValues.Floats rawVectorValues,
      IndexInput slice,
      int quantizedDimSize,
      int bquery,
      IVFRN quantizedVectorValues)
      throws IOException {
    this.rawVectorValues = rawVectorValues;
    this.slice = slice;
    this.quantizedDimSize = quantizedDimSize;
    this.quantizedByteSize = (quantizedDimSize + Integer.BYTES * 4) * quantizedVectorValues.C;
    this.quantizedVectorValues = quantizedVectorValues;
    this.bquery = bquery;
  }

  @Override
  public RandomVectorScorer scorer(int ord) throws IOException {
    slice.seek(quantizedByteSize * ord);
    IVFRN.QuantizedQuery[] quantizedQuery = new IVFRN.QuantizedQuery[quantizedVectorValues.C];
    for (int i = 0; i < quantizedVectorValues.C; i++) {
      quantizedQuery[i] = IVFRN.QuantizedQuery.readFrom(slice, quantizedDimSize, i);
    }
    if (bquery == 2) {
      return new RBQRandomVectorScorerB2(quantizedQuery, rawVectorValues, quantizedVectorValues);
    }
    return new RBQRandomVectorScorer(quantizedQuery, rawVectorValues, quantizedVectorValues);
  }

  @Override
  public RandomVectorScorerSupplier copy() throws IOException {
    return new OffHeapRBQRandomVectorScorerSupplier(
        rawVectorValues, slice.clone(), quantizedDimSize, bquery, quantizedVectorValues);
  }

  public static class RBQRandomVectorScorerB2
      extends RandomVectorScorer.AbstractRandomVectorScorer {
    // TODO do we ever need to rerank?
    private final IVFRN.QuantizedQuery[] quantizedQuery;
    private final IVFRN quantizedVectorValues;

    /**
     * Creates a new scorer for the given vector values.
     *
     * @param values the vector values
     */
    public RBQRandomVectorScorerB2(
        IVFRN.QuantizedQuery[] quantizedQuery,
        RandomAccessVectorValues values,
        IVFRN quantizedVectorValues) {
      super(values);
      this.quantizedQuery = quantizedQuery;
      this.quantizedVectorValues = quantizedVectorValues;
    }

    @Override
    public float score(int node) throws IOException {
      int centroidId = quantizedVectorValues.getCentroidId(node);
      IVFRN.QuantizedQuery quantizedQueryValue = quantizedQuery[centroidId];
      float comparison = quantizedVectorValues.quantizeCompareB2(quantizedQueryValue, node);
      return 1 / (1f + comparison);
    }
  }

  public static class RBQRandomVectorScorer extends RandomVectorScorer.AbstractRandomVectorScorer {
    // TODO do we ever need to rerank?
    private final IVFRN.QuantizedQuery[] quantizedQuery;
    private final IVFRN quantizedVectorValues;

    /**
     * Creates a new scorer for the given vector values.
     *
     * @param values the vector values
     */
    public RBQRandomVectorScorer(
        IVFRN.QuantizedQuery[] quantizedQuery,
        RandomAccessVectorValues values,
        IVFRN quantizedVectorValues) {
      super(values);
      this.quantizedQuery = quantizedQuery;
      this.quantizedVectorValues = quantizedVectorValues;
    }

    @Override
    public float score(int node) throws IOException {
      int centroidId = quantizedVectorValues.getCentroidId(node);
      IVFRN.QuantizedQuery quantizedQueryValue = quantizedQuery[centroidId];
      float comparison = quantizedVectorValues.quantizeCompare(quantizedQueryValue, node);
      return 1 / (1f + comparison);
    }
  }
}
