package org.apache.lucene.sandbox.rabitq;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;

public class RBQRandomVectorScorerSupplier implements RandomVectorScorerSupplier {

  private final RandomAccessVectorValues.Floats rawVectorValues;
  private final RandomAccessVectorValues.Floats rawVectorValues1;
  private final RandomAccessVectorValues.Floats rawVectorValues2;
  private final int B_QUERY;
  // Right now, these are on heap
  private final IVFRN quantizedVectorValues;

  public RBQRandomVectorScorerSupplier(
      RandomAccessVectorValues.Floats rawVectorValues, IVFRN quantizedVectorValues, int B_QUERY)
      throws IOException {
    this.rawVectorValues = rawVectorValues;
    this.rawVectorValues1 = rawVectorValues.copy();
    this.rawVectorValues2 = rawVectorValues.copy();
    this.quantizedVectorValues = quantizedVectorValues;
    this.B_QUERY = B_QUERY;
  }

  @Override
  public RandomVectorScorer scorer(int ord) throws IOException {
    float[] vector = rawVectorValues1.vectorValue(ord);

    IVFRN.QuantizedQuery[] quantizedQuery = quantizedVectorValues.quantizeQuery(vector, B_QUERY);
    return new RBQRandomVectorScorer(
        vector, quantizedQuery, rawVectorValues2, quantizedVectorValues, B_QUERY);
  }

  @Override
  public RandomVectorScorerSupplier copy() throws IOException {
    return new RBQRandomVectorScorerSupplier(rawVectorValues, quantizedVectorValues, B_QUERY);
  }

  public static class RBQRandomVectorScorer extends RandomVectorScorer.AbstractRandomVectorScorer {
    // TODO do we ever need to rerank?
    private final float[] queryVector;
    private final IVFRN.QuantizedQuery[] quantizedQuery;
    private final IVFRN quantizedVectorValues;
    private final int B_QUERY;

    private final Map<Byte, Integer> bitCountMap = new HashMap<>();

    /**
     * Creates a new scorer for the given vector values.
     *
     * @param values the vector values
     */
    public RBQRandomVectorScorer(
        float[] queryVector,
        IVFRN.QuantizedQuery[] quantizedQuery,
        RandomAccessVectorValues values,
        IVFRN quantizedVectorValues,
        int B_QUERY) {
      super(values);
      this.queryVector = queryVector;
      this.quantizedQuery = quantizedQuery;
      this.quantizedVectorValues = quantizedVectorValues;
      this.B_QUERY = B_QUERY;

      for (int i = 0; i < 256; i++) { // Loop through all possible byte values (0 to 255)
        int bitsSet = SpaceUtils.countBits((byte) i); // Count the number of bits set in the current byte
        bitCountMap.put((byte)i, bitsSet); // Store the mapping in the HashMap
      }
    }

    @Override
    public float score(int node) throws IOException {
      int centroidId = quantizedVectorValues.getCentroidId(node);
      IVFRN.QuantizedQuery quantizedQueryValue = quantizedQuery[centroidId];
      float comparison = quantizedVectorValues.quantizeCompare(quantizedQueryValue, node, B_QUERY, this.bitCountMap);
      // Flip so biggest value is closest
      return 1 / (1f + comparison);
    }
  }
}
