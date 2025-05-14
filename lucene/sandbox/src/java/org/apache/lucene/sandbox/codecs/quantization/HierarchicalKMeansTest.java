package org.apache.lucene.sandbox.codecs.quantization;

import org.apache.lucene.index.FloatVectorValues;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class HierarchicalKMeansTest {
  public static void main(String[] args) throws IOException {
    // FIXME: write a test that utilizes hkmeans the same as in toms code

    int targetSize = 40;
    HierarchicalKMeans hkmeans = new HierarchicalKMeans();

    int dimensions = 768;
    int vectorCount = 300;
    Random random = new Random(42L);
    List<float[]> vectors = new ArrayList<>();

    for(int v = 0; v < vectorCount; v++) {
      float[] vector = new float[dimensions];
      for (int i = 0; i < dimensions; i++) {
        vector[i] = random.nextFloat();
      }
      vectors.add(vector);
    }

    System.out.println(" ==== inputs: ");
    for(int i = 0; i < vectors.size(); i++) {
      System.out.println(Arrays.toString(vectors.get(i)));
    }

    FloatVectorValues inputVectors = FloatVectorValues.fromFloats(vectors, dimensions);
    KMeansResult result = hkmeans.cluster(inputVectors, targetSize);

    float[][] centroids = result.centroids();
    short[] assignments = result.assignments();
    short[] soarAssignments = result.soarAssignments();

    System.out.println(" ==== outputs: ");
    System.out.println(centroids.length);
    System.out.println(Arrays.deepToString(centroids));
    System.out.println(Arrays.toString(assignments));
    System.out.println(Arrays.toString(soarAssignments));
  }
}
