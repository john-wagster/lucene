package org.apache.lucene.sandbox.rabitq;

public class Centroid {

  private final int id;
  private final float[] vector;

  public Centroid(int id, float[] initialVector) {
    this.id = id;
    this.vector = initialVector;
  }

  public int getId() {
    return id;
  }

  public float[] getVector() {
    return vector;
  }
}
