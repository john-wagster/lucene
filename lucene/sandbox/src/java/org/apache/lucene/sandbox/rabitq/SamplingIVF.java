package org.apache.lucene.sandbox.rabitq;

import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class SamplingIVF {
    private static final int seed = 1;
    private static final Random rand = new Random(seed);

    private final int totalClusters;
    private Centroid[] centroids;

    // cache these for now since our only real operation is to get the trained vector distances on search
    private Map<Integer, SearchResult> vectorToCentroid;
    private Map<Integer, Set<Integer>> centroidToVectors;

    public SamplingIVF(int clusters) {
        this.totalClusters = clusters;
    }

    public void train(RandomAccessVectorValues.Floats vectors, int dimensions) throws IOException {
        // FIXME: FUTURE do error checking ... has to be at least as many vectors as centroids for now

        int vectorsLength = vectors.size();

        vectorToCentroid = new HashMap<>();
        centroidToVectors = new HashMap<>();

        // FIXME: build this random selection of of the first N vectors where N is a reasonable set to consider
        // randomly create totalClusters centroids from existing vectors
        this.centroids = new Centroid[totalClusters];
        List<Integer> randomStartVectors = getRandomAndRemove(
        IntStream.range(0, vectorsLength).boxed().collect(Collectors.toList()), totalClusters);
        for (int i = 0; i < totalClusters; i++) {
            int vectorIndex = randomStartVectors.get(i);
            float[] vector = Arrays.copyOf(vectors.vectorValue(vectorIndex), vectors.dimension());
            this.centroids[i] = new Centroid(i, vector);
            HashSet<Integer> vecs = new HashSet<>();
            vecs.add(vectorIndex);
            centroidToVectors.put(i, vecs);
            vectorToCentroid.put(vectorIndex, new SearchResult(-1f, i));
        }

        // FIXME: FUTURE - replace w logging
        System.out.println("iterating over centroid positions");

        int iterations = 0;
        int maxIterations = 5;
        while (iterations < maxIterations) {
            List<Integer> exploredVectors = getRandomAndRemove(
              IntStream.range(0, vectorsLength).boxed().collect(Collectors.toList()),
              (int) (vectorsLength * 0.10));

            centroidToVectors = new HashMap<>();


          for (int i : exploredVectors) {
            float[] vector = vectors.vectorValue(i);

            float smallestDToCentroid = Float.MAX_VALUE;
            int centroid = -1;
            for (int j = 0; j < centroids.length; j++) {
              float d = VectorUtils.squareDistance(centroids[j].getVector(), vector);
              if (d < smallestDToCentroid) {
                smallestDToCentroid = d;
                centroid = j;
              }
            }
            Set<Integer> vectorIds = centroidToVectors.getOrDefault(centroid, new HashSet<>());
            vectorIds.add(i);
            centroidToVectors.putIfAbsent(centroid, vectorIds);
          }

                // for each of the associated nearest vectors move the centroid closer to them
          for (int i = 0; i < centroids.length; i++) {
              Set<Integer> vectorIds = centroidToVectors.getOrDefault(i, new HashSet<>());

              if (vectorIds.isEmpty()) {
                  continue;
              }

              // FIXME: FUTURE - this produces a potential set of NaN vectors when no vectors are near the centroid; exclude those centroids?
              double[] sums = new double[dimensions];
              for (int vecId : vectorIds) {
                  float[] vector = vectors.vectorValue(vecId);
                  for (int a = 0; a < dimensions; a++) {
                      sums[a] += vector[a];
                  }
              }
              for (int j = 0; j < sums.length; j++) {
                  centroids[i].getVector()[j] = (float) (sums[j] / (float) vectorIds.size());
              }
              assert !Float.isNaN(centroids[i].getVector()[0]);
          }

          // FIXME: FUTURE - replace w logging
          System.out.print(".");

          iterations++;
        }
        System.out.println();
    }

    public Centroid[] getCentroids() {
        return centroids;
    }

    public SearchResult[] search(RandomAccessVectorValues.Floats vectorValues) throws IOException {

        SearchResult[] searchResults = new SearchResult[vectorValues.size()];

        for (int i = 0; i < vectorValues.size(); i++) {
            float[] vector = vectorValues.vectorValue(i);

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

    public static List<Integer> getRandomAndRemove(List<Integer> list, int totalItems) {
        List<Integer> outputList = new ArrayList<>();

        for(int i = 0; i < totalItems; i++) {
            int index = rand.nextInt(list.size());
            int selectedNumber = list.get(index);
            list.remove(index); // Remove the selected number from the list
            outputList.add(selectedNumber);
        }

        return outputList;
    }
}