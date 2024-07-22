package org.apache.lucene.sandbox.rabitq;

public record IVFRNStats(
    int maxEstimatorSize, int totalEstimatorQueueAdds, int floatingPointOps, float errorBoundAvg) {}
