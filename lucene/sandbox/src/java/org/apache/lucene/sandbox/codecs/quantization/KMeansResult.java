/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.sandbox.codecs.quantization;

public class KMeansResult {
  private float[][] centroids;
  private final short[] assignments;
  private final int[] assignmentOrds;
  private short[] soarAssignments;

  public KMeansResult(float[][] centroids, short[] assignments, int[] assignmentOrds, short[] soarAssignments) {
    this.centroids = centroids;
    this.assignments = assignments;
    this.assignmentOrds = assignmentOrds;
    this.soarAssignments = soarAssignments;
  }

  public KMeansResult(float[][] centroids, short[] assignments, int[] assignmentOrdinals) {
    this(centroids, assignments, assignmentOrdinals, null);
  }

  public KMeansResult() {
    this(new float[0][0], new short[0], new int[0], new short[0]);
  }

  public float[][] centroids() {
    return centroids;
  }

  public void setCentroids(float[][] centroids) {
    this.centroids = centroids;
  }

  public short[] assignments() {
    return assignments;
  }

  public int[] assignmentOrds() {
    return assignmentOrds;
  }

  public short[] soarAssignments() {
    return soarAssignments;
  }

  public void setSoarAssignments(short[] soarAssignments) {
    this.soarAssignments = soarAssignments;
  }

}