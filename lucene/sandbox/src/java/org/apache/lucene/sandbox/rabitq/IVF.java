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

    private static final int seed = 1;
    private static final Random rand = new Random(seed);

    private final int totalClusters;
    private Centroid[] centroids;

    // cache these for now since our only real operation is to get the trained vector distances on search
    private Map<Integer, SearchResult> vectorToCentroid;
    private Map<Integer, Set<Integer>> centroidToVectors;

    public IVF(int clusters) {
        this.totalClusters = clusters;
    }

    public void train(float[][] vectors) {
        // FIXME: FUTURE do error checking ... has to be at least as many vectors as centroids for now

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

        // iterate through all vectors until the centroids stop moving around
        boolean stable;

        // FIXME: FUTURE - replace w logging
        System.out.println("iterating over centroid positions");

        // FIXME: this is too slow
        // FIXME: FUTURE - cleaner loop
        int iterations = 0;
        int maxIterations = 5;
        while (true) {
            stable = true;

            for(int i = 0; i < vectors.length; i++) {
                SearchResult activeVecMetadata = vectorToCentroid.getOrDefault(i, new SearchResult(-1, -1));
                int priorCentroid = activeVecMetadata.getClusterId();
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
                activeVecMetadata.setDistToCentroid(smallestDToCentroid);
                if(vectorIds.add(i)) {
                    stable = false;
                    if(priorCentroid != -1) {
                        centroidToVectors.get(priorCentroid).remove(i);
                    }
                    activeVecMetadata.setClusterId(centroid);
                    vectorToCentroid.put(i, activeVecMetadata);
                }
            }

            if( stable ) {
                break;
            }

            // for each of the associated nearest vectors move the centroid closer to them
            for(int i = 0; i < centroids.length; i++) {
                Set<Integer> vectorIds = centroidToVectors.get(i);

                if(vectorIds.isEmpty()) {
                    continue;
                }

                // FIXME: FUTURE - this produces a potential set of NaN vectors when no vectors are near the centroid; exclude those centroids?
                int dimensions = vectors[0].length;
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
            if(iterations >= maxIterations) {
                break;
            }
        }
    }

    public Centroid[] getCentroids() {
        return centroids;
    }

    public SearchResult[] getTrainedVectorCentroid(float[][] vectors) {
        // relies on the assumption that vectors is in the same order it was sent for training
        int size = this.vectorToCentroid.size();
        SearchResult[] searchResults = new SearchResult[size];
        for(int i = 0; i < size; i++) {
            SearchResult res = this.vectorToCentroid.get(i);
            if (res.getDistToCentroid() < 0) {
                float d = VectorUtils.squareDistance(centroids[res.getClusterId()].getVector(), vectors[i]);
                res.setDistToCentroid(d);
            }
            searchResults[i] = res;
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
