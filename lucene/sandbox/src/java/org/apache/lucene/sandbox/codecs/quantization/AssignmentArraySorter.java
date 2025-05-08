/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.sandbox.codecs.quantization;

import org.apache.lucene.util.TimSorter;

class AssignmentArraySorter extends TimSorter {

  private final int[] arr;
  private final int[] tmp;

  private final float[] distances;
  private final float[] tmpd;

  public AssignmentArraySorter(int[] docIds, float[] distances) {
    super(docIds.length / 64);
    this.arr = docIds;
    int maxTempSlots = arr.length / 64;
    this.distances = distances;
    this.tmp = new int[maxTempSlots];
    this.tmpd = new float[maxTempSlots];
  }

  @Override
  protected int compare(int i, int j) {
    return Float.compare(distances[i], distances[j]);
  }

  @Override
  protected void swap(int i, int j) {
    final int tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;

    final float tmpd = distances[i];
    distances[i] = distances[j];
    distances[j] = tmpd;
  }

  @Override
  protected void copy(int src, int dest) {
    arr[dest] = arr[src];
    distances[dest] = distances[src];
  }

  @Override
  protected void save(int start, int len) {
    System.arraycopy(arr, start, tmp, 0, len);
    System.arraycopy(distances, start, tmpd, 0, len);
  }

  @Override
  protected void restore(int src, int dest) {
    arr[dest] = tmp[src];
    distances[dest] = tmpd[src];
  }

  @Override
  protected int compareSaved(int i, int j) {
    return Float.compare(tmpd[i], distances[j]);
  }
}