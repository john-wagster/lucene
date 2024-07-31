package org.apache.lucene.sandbox.rabitq;

import org.apache.lucene.util.VectorUtil;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Comparator;
import java.util.TreeSet;

public class BruteForce {

  private static final int KNN = 10;

  record Pair(Float distance, Integer index) {}

  public static void main(String[] args) throws IOException {
    float[][] queries = IOUtils.readFvecs(new FileInputStream(args[0]));
    float[][] corpus = IOUtils.readFvecs(new FileInputStream(args[1]));

    int iterations = 0;
    int[][] groundTruth = new int[queries.length][KNN];
    for (int i = 0; i < 50; i++) {
      TreeSet<Pair> topKNNDistances =
          new TreeSet<>(
                  (o1, o2) -> {
                    if (o1.distance > o2.distance) {
                      return -1;
                    } else {
                      return 1;
                    }
                  });
      boolean filledKNN = false;

//      float furtherestTopKNN = Float.MAX_VALUE;
      float furtherestTopKNN = Float.MIN_VALUE;
      for (int j = 0; j < corpus.length; j++) {
//        float distance = VectorUtils.squareDistance(queries[i], corpus[j]);
//        float distance = VectorUtil.dotProduct(queries[i], corpus[j]);
        float distance = VectorUtil.scaleMaxInnerProductScore(VectorUtil.dotProduct(queries[i], corpus[j]));
//        if (distance < furtherestTopKNN) {
        if (distance > furtherestTopKNN) {
          topKNNDistances.add(new Pair(distance, j));
          if (!filledKNN) {
            if (topKNNDistances.size() == KNN) {
              furtherestTopKNN = topKNNDistances.last().distance;
              filledKNN = true;
            }
          } else {
            topKNNDistances.removeLast();
            furtherestTopKNN = topKNNDistances.last().distance;
          }
        }
      }

      groundTruth[i] = topKNNDistances.stream().mapToInt(a -> a.index).toArray();

      if (iterations % 100 == 0) {
        System.out.print(".");
      }
      iterations++;
    }

    IOUtils.toIvecs(new FileOutputStream(args[2]), groundTruth);
  }
}
