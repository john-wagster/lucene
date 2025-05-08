/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.sandbox.codecs.quantization;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;

import java.io.IOException;

public class HierarchicalKMeans {

  static final int MAXK = 128;

  final int maxIterations;
  final int samplesPerCluster;
  final short clustersPerNeighborhood;

  public HierarchicalKMeans() {
    this(8, 256, (short) MAXK);
  }

  public HierarchicalKMeans(final int maxIterations,
                            final int samplesPerCluster,
                            final short clustersPerNeighborhood) {
    this.maxIterations = maxIterations;
    this.samplesPerCluster = samplesPerCluster;
    this.clustersPerNeighborhood = clustersPerNeighborhood;
  }

  public KMeansResult cluster(FieldInfo fieldInfo, FloatVectorValues vectors, int desiredClusters) throws IOException {
    int targetSize = (int) (vectors.size() / (float) desiredClusters);
//    int targetSize = (int) (desiredClusters * 0.33f);

    KMeansResult kMeansResult = kMeansHierarchical(fieldInfo, new FloatVectorValuesSlice(vectors), targetSize, maxIterations, samplesPerCluster);

    if (kMeansResult.centroids().length > 1 && kMeansResult.centroids().length < vectors.size()) {
//      long startTimeLocalKmeans = System.nanoTime();

      KMeansLocal.kMeansLocal(vectors, kMeansResult, clustersPerNeighborhood, maxIterations);

      // FIXME: remove me
//      System.out.println(" ==== local kmeans ms: " + (System.nanoTime() - startTimeLocalKmeans) / 1000000.0);
    }

    return kMeansResult;

  }

  static KMeansResult kMeansHierarchical(final FieldInfo fieldInfo,
                                         final FloatVectorValuesSlice vectors,
                                         final int targetSize,
                                         final int maxIterations,
                                         final int samplesPerCluster) throws IOException {
    if (vectors.size() <= targetSize) {
      return new KMeansResult();
    }

    int k = Math.clamp((int)((vectors.size() + targetSize / 2.0f) / (float) targetSize), 2, MAXK);
    int m = Math.min(k * samplesPerCluster, vectors.size());

    // FIXME: get rid of the recursion and when you do get rid of these as well and just reference the "parent" depth=0 level arrays only
    short[] assignments = new short[vectors.size()];

    long startTime = System.nanoTime();

    final KMeans.Results kMeans =
      KMeans.cluster(
        vectors,
        k,
        false,
        42L,
        KMeans.KmeansInitializationMethod.FORGY,
        null,
        fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.COSINE,
        1,
        maxIterations,
        m);
    float[][] centroids = kMeans.centroids();

    // FIXME: remove me
//    System.out.println(" ==== kmeans ms: " + (System.nanoTime() - startTime) / 1000000.0);

    int[] clusterSizes = new int[centroids.length];

    long startTimeKmeans = System.nanoTime();

    // FIXME: consider adding cluster size counts to the kmeans algo?
    // handle assignment here so we can track distance and cluster size
    for(int i = 0; i < vectors.size(); i++) {
      float smallest = Float.MAX_VALUE;
      short centroidIdx = -1;
      float[] vector = vectors.vectorValue(i);
      for (short j = 0; j < centroids.length; j++) {
        float[] centroid = centroids[j];
        float d = VectorUtil.squareDistance(vector, centroid);
        if (d < smallest) {
          smallest = d;
          centroidIdx = j;
        }
      }
      assignments[i] = centroidIdx;
      clusterSizes[centroidIdx]++;
    }

    // FIXME: remove me
//    System.out.println(" ==== assignment ms: " + (System.nanoTime() - startTimeKmeans) / 1000000.0);

    short effectiveK = 0;
    for(int i = 0; i < clusterSizes.length; i++) {
      if(clusterSizes[i] > 0) {
        effectiveK++;
      }
    }

    int[] assignmentOrdinals = new int[vectors.slice.length];
    for(int i = 0; i < assignmentOrdinals.length; i++) {
      assignmentOrdinals[i] = vectors.slice[i];
    }

    KMeansResult kMeansResult = new KMeansResult(centroids, assignments, assignmentOrdinals);

    if (effectiveK == 1) {
      return kMeansResult;
    }

    for (short c = 0; c < clusterSizes.length; c++) {
      // Recurse for each cluster which is larger than targetSize
      // Give ourselves 30% margin for the target size
      if (100 * clusterSizes[c] > 134 * targetSize) {
        FloatVectorValuesSlice sample = createClusterSlice(clusterSizes[c], c, vectors, assignments);

        // FIXME: rewrite this without recursion and keep a stack of the fvv slices
        updateAssignmentsWithRecursiveSplit(
          kMeansResult, c, kMeansHierarchical(
            fieldInfo, sample, targetSize,
            maxIterations, samplesPerCluster
          )
        );
      }
    }

    return kMeansResult;
  }

  static FloatVectorValuesSlice createClusterSlice(int clusterSize, int cluster, FloatVectorValuesSlice vectors, short[] assignments) {
    int[] slice = new int[clusterSize];
    int idx = 0;
    for(int i = 0; i < assignments.length; i++) {
      if(assignments[i] == cluster) {
        slice[idx] = i;
        idx++;
      }
    }

    return new FloatVectorValuesSlice(vectors, slice);
  }

  static void updateAssignmentsWithRecursiveSplit(KMeansResult current, short cluster, KMeansResult splitClusters) {
    int orgCentroidsSize = current.centroids().length;

    // update based on the outcomes from the split clusters recursion
    if(splitClusters.centroids().length > 1) {
      float[][] newCentroids = new float[current.centroids().length +
        splitClusters.centroids().length - 1][current.centroids()[0].length];
      System.arraycopy(current.centroids(), 0, newCentroids, 0, current.centroids().length);

      // replace the original cluster
      short origCentroidOrd = 0;
      newCentroids[cluster] = splitClusters.centroids()[0];

      // append the remainder
      System.arraycopy(splitClusters.centroids(), 1, newCentroids, current.centroids().length, splitClusters.centroids().length-1);

      current.setCentroids(newCentroids);

      for(int i = 0; i < splitClusters.assignments().length; i++) {
        // this is a new centroid that was added, and so we'll need to remap it
        if(splitClusters.assignments()[i] != origCentroidOrd) {
          int parentOrd = splitClusters.assignmentOrds()[i];
          assert current.assignments()[parentOrd] == cluster;
          current.assignments()[parentOrd] = (short) (splitClusters.assignments()[i] + orgCentroidsSize - 1);
        }
      }
    }
  }

}
