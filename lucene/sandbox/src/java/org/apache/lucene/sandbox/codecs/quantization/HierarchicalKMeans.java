/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.sandbox.codecs.quantization;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.util.VectorUtil;

import java.io.IOException;

public class HierarchicalKMeans {

  static final int MAXK = 128;
  static final int MAX_ITERATIONS_DEFAULT = 6;
  static final int SAMPLES_PER_CLUSTER_DEFAULT = 256;

  final int maxIterations;
  final int samplesPerCluster;
  final short clustersPerNeighborhood;

  public HierarchicalKMeans() {
    this(MAX_ITERATIONS_DEFAULT, SAMPLES_PER_CLUSTER_DEFAULT, (short) MAXK);
  }

  public HierarchicalKMeans(int maxIterations,
                            int samplesPerCluster,
                            short clustersPerNeighborhood) {
    this.maxIterations = maxIterations;
    this.samplesPerCluster = samplesPerCluster;
    this.clustersPerNeighborhood = clustersPerNeighborhood;
  }

  public KMeansResult cluster(FloatVectorValues vectors, int targetSize) throws IOException {
    KMeansResult kMeansResult = kMeansHierarchical(new FloatVectorValuesSlice(vectors), targetSize);

    if (kMeansResult.centroids().length > 1 && kMeansResult.centroids().length < vectors.size()) {
      // FIXME: should we do the same number of iterations here for minimally simplicity
      KMeansLocal.kMeansLocal(vectors, kMeansResult, clustersPerNeighborhood, 8);
    }

    return kMeansResult;

  }

//  bool stepLloyd(std::size_t nd,
//                 std::size_t dim,
//               const Dataset& dataset,
//                 Centers& centers,
//                 Centers& nextCenters,
//                 std::vector<std::size_t>& q,
//                 std::vector<std::size_t>& a) {
//
//    bool changed{false};
//
//    nextCenters.assign(centers.size(), 0.0F);
//    q.assign(centers.size() / dim, 0);
//
//    for (std::size_t i = 0, id = 0; id < nd; ++i, id += dim) {
//      std::size_t bestJd{0};
//      float minDsq{INF};
//      for (std::size_t jd = 0; jd < centers.size(); jd += dim) {
//        float dsq{distanceSq(dim, &dataset[id], &centers[jd])};
//        if (dsq < minDsq) {
//          minDsq = dsq;
//          bestJd = jd;
//        }
//      }
//      changed |= (a[i] != bestJd);
//      a[i] = bestJd;
//      ++q[bestJd / dim];
//        #pragma omp simd
//      for (std::size_t d = 0; d < dim; ++d) {
//        nextCenters[bestJd + d] += dataset[id + d];
//      }
//    }

  KMeansResult kMeansHierarchical(final FloatVectorValuesSlice vectors,
                                         final int targetSize) throws IOException {
    if (vectors.size() <= targetSize) {
      return new KMeansResult();
    }

    int k = Math.clamp((int)((vectors.size() + targetSize / 2.0f) / (float) targetSize), 2, MAXK);
    int m = Math.min(k * samplesPerCluster, vectors.size());

    // TODO: instead of creating a sub-cluster assignments reuse the parent array each time
    short[] assignments = new short[vectors.size()];

    final KMeans.Results kMeans =
      KMeans.cluster(
        vectors,
        k,
        false,
        42L,
        KMeans.KmeansInitializationMethod.FORGY,
        null,
        false,
        1,
        maxIterations,
        m);
    float[][] centroids = kMeans.centroids();

    int[] clusterSizes = new int[centroids.length];

    // TODO: consider adding cluster size counts to the kmeans algo
    // handle assignment here so we can track distance and cluster size
    int[] centroidVectorCount = new int[centroids.length];
    float[][] nextCentroids = new float[centroids.length][vectors.dimension()];
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
      centroidVectorCount[centroidIdx]++;
      for(int j = 0; j < vectors.dimension(); j++) {
        nextCentroids[centroidIdx][j] += vector[j];
      }
      assignments[i] = centroidIdx;
      clusterSizes[centroidIdx]++;
    }

    // update centroids based on assignments of all vectors
    for(int i = 0; i < centroids.length; i++) {
      if(centroidVectorCount[i] > 0) {
        for(int j = 0; j < vectors.dimension(); j++) {
          centroids[i][j] = nextCentroids[i][j] / centroidVectorCount[i];
        }
      }
    }

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

        // TODO: consider iterative here instead of recursive
        updateAssignmentsWithRecursiveSplit(
          kMeansResult, c,
          kMeansHierarchical(sample, targetSize)
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
