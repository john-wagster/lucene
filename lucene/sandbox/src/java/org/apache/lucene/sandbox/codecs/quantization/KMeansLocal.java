/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.sandbox.codecs.quantization;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.NeighborQueue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public final class KMeansLocal {

  KMeansLocal() {
    // FIXME: move class level params out of kmeanslocal function
  }

  private static void computeNeighborhoods(float[][] centers,
                                           List<int[]> neighborhoods, // Modified in place
                                           int clustersPerNeighborhood) {
    int k = neighborhoods.size();

    if (k == 0 || clustersPerNeighborhood <= 0) {
      return;
    }

    List<NeighborQueue> neighborQueues = new ArrayList<>(k);
    for (int i = 0; i < k; i++) {
      neighborQueues.add(new NeighborQueue(clustersPerNeighborhood, true));
    }
    for (int i = 0; i < k - 1; i++) {
      for (int j = i + 1; j < k; j++) {
        float dsq = VectorUtil.squareDistance(centers[i], centers[j]);
        neighborQueues.get(j).insertWithOverflow(i, dsq);
        neighborQueues.get(i).insertWithOverflow(j, dsq);
      }
    }

    for (int i = 0; i < k; i++) {
      NeighborQueue queue = neighborQueues.get(i);
      int neighborCount = queue.size();
      int[] neighbors = new int[neighborCount];
      queue.consumeNodes(neighbors);
      Arrays.sort(neighbors);
      neighborhoods.set(i, neighbors);
    }
  }

  private static boolean stepLloyd(FloatVectorValues dataset,
                                   List<int[]> neighborhoods,
                                   float[][] centers,
                                   float[][] nextCenters,
                                   long[] centerCounts,
                                   short[] assignments) throws IOException {

    boolean changed = false;
    int dim = centers[0].length;
    int k = centerCounts.length;
    int n = assignments.length;

    Arrays.fill(centerCounts, 0L);
    for(int i = 0; i < nextCenters.length; i++) {
      for(int j = 0; j < nextCenters[0].length; j++) {
        nextCenters[i][j] = 0.0f;
      }
    }

    for (int i = 0; i < n; i++) {
      float[] vector = dataset.vectorValue(i);
      short currentClusterIndex = assignments[i];
      int bestCenterOffset = currentClusterIndex;

      float minDsq = VectorUtil.squareDistance(vector, centers[currentClusterIndex]);

      if (currentClusterIndex < neighborhoods.size()) {
        int[] neighborOffsets = neighborhoods.get(currentClusterIndex);
        if (neighborOffsets != null) {
          for (int neighborOffset : neighborOffsets) {
            if (neighborOffset >= 0 && neighborOffset <= centers.length) {
              float dsq = VectorUtil.squareDistance(vector, centers[neighborOffset]);
              if (dsq < minDsq) {
                minDsq = dsq;
                bestCenterOffset = neighborOffset;
              }
            }
          }
        }
      }
      if (assignments[i] != bestCenterOffset) {
        changed = true;
      }
      assignments[i] = (short) bestCenterOffset;

      // FIXME: always true?
      if (bestCenterOffset >= 0 && bestCenterOffset <= centers.length) {
        centerCounts[bestCenterOffset]++;
        for (short d = 0; d < dim; d++) {
          nextCenters[bestCenterOffset][d] += vector[d];
        }
      }
    }

    for (int clusterIdx = 0; clusterIdx < k; clusterIdx++) {
      if (centerCounts[clusterIdx] > 0) {
        float countF = (float) centerCounts[clusterIdx];
        for (int d = 0; d < dim; d++) {
          centers[clusterIdx][d] = nextCenters[clusterIdx][d] / countF;
        }
      }
    }

    return changed;
  }

  static short[] assignSpilled(FloatVectorValues vectors, List<int[]> neighborhoods,
                            float[][] centers, short[] assignments) throws IOException {
    // SOAR uses an adjusted distance for assigning spilled documents which is
    // given by:
    //
    //   soar(x, c) = ||x - c||^2 + lambda * ((x - c_1)^t (x - c))^2 / ||x - c_1||^2
    //
    // Here, x is the document, c is the nearest centroid, and c_1 is the first
    // centroid the document was assigned to. The document is assigned to the
    // cluster with the smallest soar(x, c).

    short[] spilledAssignments = new short[assignments.length];

    float[] d1 = new float[vectors.dimension()];
    for(int i = 0; i < vectors.size(); i++) {
      float[] xi = vectors.vectorValue(i);

      short currJd = assignments[i];
      float[] c1 = centers[currJd];
      for (int j = 0; j < vectors.dimension(); j++) {
        float diff = xi[j] - c1[j];
        d1[j] = diff;
      }

      // FIXME: cache these?
//      float d1sq = assignmentDistances[i];
      float d1sq = VectorUtil.squareDistance(xi, c1);

      int bestJd = -1;
      float minSoar = Float.MAX_VALUE;
      for(int jd : neighborhoods.get(currJd)) {
        if (jd == currJd) {
          continue;
        }
        float[] cj = centers[jd];
        float soar = distanceSoar(d1, xi, cj, d1sq);
        if(soar < minSoar) {
          bestJd = jd;
          minSoar = soar;
        }
      }

      spilledAssignments[i] = (short) bestJd;
    }

    return spilledAssignments;
  }

  static float distanceSoar(float[] r, float[] x, float[] c, float rnorm) {
    float lambda = 1.0F;
    // FIXME: can probably combine these to be more efficient
    float dsq = VectorUtil.squareDistance(x, c);
    float rproj = VectorUtil.soarResidual(x, c, r);
    return dsq + lambda * rproj * rproj / rnorm;
  }

  public static KMeansResult kMeansLocal(FloatVectorValues dataset,
                                         KMeansResult kMeansResult,
                                             short clustersPerNeighborhood,
                                             int maxIterations) throws IOException {
    final float[][] centroids = kMeansResult.centroids();
    final short[] assignments = kMeansResult.assignments();
    int k = centroids.length;

    List<int[]> neighborhoods = new ArrayList<>(k);
    for(int i=0; i < k; ++i) {
      neighborhoods.add(null);
    }

    computeNeighborhoods(centroids, neighborhoods, clustersPerNeighborhood);

    long[] centroidCounts = new long[k];
    float[][] nextCenters = new float[centroids.length][centroids[0].length];

    int iterationsRun;
    for (iterationsRun = 0; iterationsRun < maxIterations; iterationsRun++) {
      boolean changed = stepLloyd(dataset, neighborhoods, centroids, nextCenters, centroidCounts, assignments);
      if (!changed) {
        break;
      }
    }

    kMeansResult.setSoarAssignments(assignSpilled(dataset, neighborhoods, centroids, assignments));

    return kMeansResult;
  }
}