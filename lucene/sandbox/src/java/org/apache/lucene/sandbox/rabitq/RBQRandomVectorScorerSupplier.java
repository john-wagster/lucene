package org.apache.lucene.sandbox.rabitq;

import java.io.IOException;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;

public class RBQRandomVectorScorerSupplier implements RandomVectorScorerSupplier {

  private final RandomAccessVectorValues.Floats rawVectorValues;
  private final RandomAccessVectorValues.Floats rawVectorValues1;
  // Right now, these are on heap, but should be off heap, eventually
  private final IVFRN quantizedVectorValues;

  public RBQRandomVectorScorerSupplier(
      RandomAccessVectorValues.Floats rawVectorValues, IVFRN quantizedVectorValues)
      throws IOException {
    this.rawVectorValues = rawVectorValues;
    this.rawVectorValues1 = rawVectorValues.copy();
    this.quantizedVectorValues = quantizedVectorValues;
  }

  @Override
  public RandomVectorScorer scorer(int ord) throws IOException {
    float[] vector = rawVectorValues1.vectorValue(ord);

    IVFRN.QuantizedQuery[] quantizedQuery = quantizedVectorValues.quantizeQuery(vector);
    return new RBQRandomVectorScorer(quantizedQuery, rawVectorValues, quantizedVectorValues);
  }

  @Override
  public RandomVectorScorerSupplier copy() throws IOException {
    return new RBQRandomVectorScorerSupplier(rawVectorValues, quantizedVectorValues);
  }

  public static class RBQRandomVectorScorer extends RandomVectorScorer.AbstractRandomVectorScorer {
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
      // Flip so biggest value is closest
      return 1 / (1f + comparison);
    }
  }
}
