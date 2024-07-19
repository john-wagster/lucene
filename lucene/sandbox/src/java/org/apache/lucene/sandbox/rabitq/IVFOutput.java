package org.apache.lucene.sandbox.rabitq;

public record IVFOutput(float[] distToCentroids, int[] clusterIds, float[][] centroidVectors) {}
