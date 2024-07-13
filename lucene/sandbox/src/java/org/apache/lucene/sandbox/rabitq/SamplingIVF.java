package org.apache.lucene.sandbox.rabitq;

import java.util.ArrayList;
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

    public void train(float[][] vectors) {
        // FIXME: FUTURE do error checking ... has to be at least as many vectors as centroids for now

        int dimensions = vectors[0].length;

        vectorToCentroid = new HashMap<>();
        centroidToVectors = new HashMap<>();

        // FIXME: build this random selection of of the first N vectors where N is a reasonable set to consider
        // randomly create totalClusters centroids from existing vectors
        this.centroids = new Centroid[totalClusters];
        List<Integer> randomStartVectors = getRandomAndRemove(
                IntStream.range(0, vectors.length).boxed().collect(Collectors.toList()),
                totalClusters);
        for (int i = 0; i < totalClusters; i++) {
            int vectorIndex = randomStartVectors.get(i);
            this.centroids[i] = new Centroid(i, vectors[vectorIndex].clone());
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
                    IntStream.range(0, vectors.length).boxed().collect(Collectors.toList()),
                    (int) (vectors.length * 0.10));

            centroidToVectors = new HashMap<>();

            // FIXME: reintroduce idea of stable state and quit early
            for(int q = 0; q < exploredVectors.size(); q++) {
                int i = exploredVectors.get(q);

                float smallestDToCentroid = Float.MAX_VALUE;
                int centroid = -1;
                for(int j = 0; j < centroids.length; j++) {
                    float d = VectorUtils.squareDistance(centroids[j].getVector(), vectors[i]);
                    if(d < smallestDToCentroid) {
                        smallestDToCentroid = d;
                        centroid = j;
                    }
                }
                Set<Integer> vectorIds = centroidToVectors.getOrDefault(centroid, new HashSet<>());
                vectorIds.add(i);
                centroidToVectors.putIfAbsent(centroid, vectorIds);
            }

            // for each of the associated nearest vectors move the centroid closer to them
            for(int i = 0; i < centroids.length; i++) {
                Set<Integer> vectorIds = centroidToVectors.getOrDefault(i, new HashSet<>());

                if(vectorIds.isEmpty()) {
                    continue;
                }

                // FIXME: FUTURE - this produces a potential set of NaN vectors when no vectors are near the centroid; exclude those centroids?
                double[] sums = new double[dimensions];
                for (int vecId : vectorIds) {
                    for (int a = 0; a < dimensions; a++) {
                        sums[a] += vectors[vecId][a];
                    }
                }
                for(int j = 0; j < sums.length; j++) {
                    centroids[i].getVector()[j] = (float) (sums[j] / (float) vectorIds.size());
                }
                assert !Float.isNaN(centroids[i].getVector()[0]);
            }

            // FIXME: FUTURE - replace w logging
            System.out.print(".");

            iterations++;
        }
    }

    public Centroid[] getCentroids() {
        return centroids;
    }

    public SearchResult[] search(float[][] vectors) {
        SearchResult[] searchResults = new SearchResult[vectors.length];

        for (int i = 0; i < vectors.length; i++) {
            float smallestDToCentroid = Float.MAX_VALUE;
            int centroid = -1;
            for (int j = 0; j < centroids.length; j++) {
                float d = VectorUtils.squareDistance(centroids[j].getVector(), vectors[i]);
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
