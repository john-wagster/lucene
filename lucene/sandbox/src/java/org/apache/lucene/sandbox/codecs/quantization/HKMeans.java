package org.apache.lucene.sandbox.codecs.quantization;


import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.util.VectorUtil;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class HKMeans {

  public float[][] pickInitialCentroids(FloatVectorValues vectors, int sampleSize, int k) throws IOException {
    // Choose data points as random ensuring we have distinct points where possible
    List<Integer> candidates = new ArrayList<>(sampleSize);
    for (int i = 0; i < sampleSize; i++) {
      candidates.add(i);
    }
    Collections.shuffle(candidates, new Random(42L));

    float[][] centroids = new float[k][vectors.dimension()];
    int centroidIdx = 0;
    for(int i = 0; i < candidates.size() && centroidIdx < k; i++) {
      int cand = candidates.get(i);
      float[] vector = vectors.vectorValue(cand);
      boolean goodCandidate = true;
      if(((candidates.size() - i) - (k - centroidIdx)) > 0) {
        for (int j = 0; j < centroidIdx; j++) {
          if (!(VectorUtil.squareDistance(vector, centroids[j]) > 0.0f)) {
            goodCandidate = false;
            break;
          }
        }
      }
      if(goodCandidate) {
        System.arraycopy(vector, 0, centroids[centroidIdx], 0, vector.length);
        centroidIdx++;
      }
    }
    return centroids;
  }

  // FIXME: reconcile this with the local kmeans variant and the KMeans class
  private static boolean stepLloyd(FloatVectorValues dataset,
                                   float[][] centroids,
                                   short[] assignments,
                                   int sampleSize) throws IOException {
    boolean changed = false;
    int dim = dataset.dimension();

    long[] centroidCounts = new long[centroids.length];
    float[][] nextCenters = new float[centroids.length][dim];

    for (int i = 0; i < sampleSize; i++) {
      float[] vector = dataset.vectorValue(i);
      short bestCentroidOffset = -1;
      float minDsq = Float.MAX_VALUE;
      for (short j = 0; j < centroids.length; j++) {
        float dsq = VectorUtil.squareDistance(vector, centroids[j]);
        if (dsq < minDsq) {
          minDsq = dsq;
          bestCentroidOffset = j;
        }
      }
      if (assignments[i] != bestCentroidOffset) {
        changed = true;
      }
      assignments[i] = bestCentroidOffset;
      centroidCounts[bestCentroidOffset]++;
      for (short d = 0; d < dim; d++) {
        nextCenters[bestCentroidOffset][d] += vector[d];
      }
    }

    for (int clusterIdx = 0; clusterIdx < centroids.length; clusterIdx++) {
      if (centroidCounts[clusterIdx] > 0) {
        float countF = (float) centroidCounts[clusterIdx];
        for (int d = 0; d < dim; d++) {
          centroids[clusterIdx][d] = nextCenters[clusterIdx][d] / countF;
        }
      }
    }

    return changed;
  }

  void cluster(FloatVectorValues vectors, int sampleSize, float[][] centroids, int maxIterations) throws IOException {
    int k = centroids.length;
    int n = vectors.size();

    if (k == 1 || k >= n) {
      return;
    }

    short[] assignments = new short[n];

    for (int i = 0; i < maxIterations; i++) {
      if (!stepLloyd(vectors, centroids, assignments, sampleSize)) {
        break;
      }
    }
    stepLloyd(vectors, centroids, assignments, vectors.size());
  }
}
