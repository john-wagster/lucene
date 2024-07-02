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

public class IVF {

    private static int seed = 1;
    private static Random rand = new Random(seed);

    private int totalClusters;
    private Centroid[] centroids;

    public IVF(int clusters) {
        this.totalClusters = clusters;
    }

    public void train(float[][] vectors) {
        // FIXME: FUTURE do error checking ... has to be at least as many vectors as centroids for now

        // randomly create totalClusters centroids from existing vectors
        this.centroids = new Centroid[totalClusters];
        List<Integer> randomStartVectors = getRandomAndRemove(
                IntStream.range(0, vectors.length).boxed().collect(Collectors.toList()),
                totalClusters);
        for (int i = 0; i < totalClusters; i++) {
            int vectorIndex = randomStartVectors.get(i);
            this.centroids[i] = new Centroid(i, vectors[vectorIndex].clone());
        }

        Random random = new Random(1);
        for(int i = 0; i < this.centroids.length; i++) {
            for(int j = 0; j < this.centroids[0].getVector().length; j++) {
                this.centroids[0].getVector()[j] = this.centroids[0].getVector()[j] + random.nextFloat(-0.001f, 0.0011f);
            }
        }

        // iterate through all vectors until the centroids stop moving around
        boolean stable = false;
        float epsilon = 0.01f;

        // FIXME: FUTURE - replace w logging
        System.out.println("iterating over centroid positions");
        Map<Integer, Integer> vectorToCentroid = new HashMap<>();
        Map<Integer, Set<Integer>> centroidToVectors = new HashMap<>();
        for(int i = 0; i < centroids.length; i++) {
            centroidToVectors.put(i, new HashSet<>());
        }

        // FIXME: this is too slow
        // FIXME: FUTURE - cleaner loop
        int iterations = 0;
        int maxIterations = 1000;
        while (true) {
            stable = true;

            for(int i = 0; i < vectors.length; i++) {
                int priorCentroid = vectorToCentroid.getOrDefault(i, -1);
                float smallestDToCentroid = Float.MAX_VALUE;
                int centroid = -1;
                for(int j = 0; j < centroids.length; j++) {
                    float d = VectorUtils.squareDistance(centroids[j].getVector(), vectors[i]);
                    if(d < smallestDToCentroid) {
                        smallestDToCentroid = d;
                        centroid = j;
                    }
                }
                Set<Integer> vectorIds = centroidToVectors.get(centroid);
                if(vectorIds.add(i)) {
                    stable = false;
                    if(priorCentroid != -1) {
                        centroidToVectors.get(priorCentroid).remove(i);
                    }
                    vectorToCentroid.put(i, centroid);
                }
            }

            if( stable ) {
                break;
            }

            // for each of the associated nearest vectors move the centroid closer to them
            for(int i = 0; i < centroids.length; i++) {
                Set<Integer> vectorIds = centroidToVectors.get(i);

                // FIXME: FUTURE - this produces a potential set of NaN vectors when no vectors are near the centroid; exclude those centroids?
                int dimensions = vectors[0].length;
                double[] sums = new double[dimensions];
                for (int j = 0; j < vectorIds.size(); j++) {
                    for (int a = 0; a < dimensions; a++) {
                        sums[a] += vectors[j][a];
                    }
                }
                for(int j = 0; j < sums.length; j++) {
                    centroids[i].getVector()[j] = (float) (sums[j] / (float) vectorIds.size());
                }
                assert centroids[i].getVector()[0] == Float.NaN;
            }

            if(iterations % 150 == 0) {
                System.out.print(".");
            }
            iterations++;

            if(iterations >= maxIterations) {
                break;
            }
        }
    }

    public Centroid[] getCentroids() {
        return centroids;
    }

    public SearchResult[] search(float[][] vectors) {
        // FIXME: FUTURE - knn instead of 1nn
        // FIXME: FUTURE - dedup this logic from above in train function
        SearchResult[] searchResults = new SearchResult[vectors.length];

        for (int i = 0; i < vectors.length; i++) {
            float smallestDToCentroid = Float.MAX_VALUE;
            int centroid = -1;
            for (int j = 0; j < centroids.length; j++) {
                // FIXME: FUTURE - replace all instances of this with the VectorUtil using Panama?? (internal VectorUtil)
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
