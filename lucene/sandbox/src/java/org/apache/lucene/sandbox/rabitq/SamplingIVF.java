package org.apache.lucene.sandbox.rabitq;

import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

import java.io.IOException;

import static org.apache.lucene.sandbox.rabitq.Index.COARSE_CLUSTERING_KMEANS_ITR;
import static org.apache.lucene.sandbox.rabitq.Index.COARSE_CLUSTERING_KMEANS_RESTARTS;

public class SamplingIVF {
    private static final int seed = 1;

    private final int totalClusters;
    private Centroid[] centroids;

    public SamplingIVF(int clusters) {
        this.totalClusters = clusters;
    }

    public void train(RandomAccessVectorValues.Floats vectors, int dimensions) throws IOException {
        // FIXME: build this random selection of of the first N vectors where N is a reasonable set to consider
        // randomly create totalClusters centroids from existing vectors
        this.centroids = new Centroid[totalClusters];
        KMeans kmeans = new KMeans(vectors, totalClusters, seed);
        float[][] centroids = kmeans.computeCentroids(COARSE_CLUSTERING_KMEANS_RESTARTS, COARSE_CLUSTERING_KMEANS_ITR, v -> {});
        for (int i = 0; i < totalClusters; i++) {
            this.centroids[i] = new Centroid(i, centroids[i]);
        }
    }

    public Centroid[] getCentroids() {
        return centroids;
    }

    public SearchResult[] search(RandomAccessVectorValues.Floats vectors) throws IOException {
        int vectorsLength = vectors.size();

        SearchResult[] searchResults = new SearchResult[vectorsLength];

        for (int i = 0; i < vectorsLength; i++) {
            float[] vector = vectors.vectorValue(i);
            float smallestDToCentroid = Float.MAX_VALUE;
            int centroid = -1;
            for (int j = 0; j < centroids.length; j++) {
                float d = VectorUtils.squareDistance(centroids[j].getVector(), vector);
                if( d < smallestDToCentroid) {
                    smallestDToCentroid = d;
                    centroid = j;
                }
            }
            searchResults[i] = new SearchResult(smallestDToCentroid, centroids[centroid].getId());
        }
        return searchResults;
    }
}
