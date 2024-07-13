package org.apache.lucene.sandbox.rabitq;

public record IVFRNStats(int maxEstimatorSize, int totalEstimatorQueueAdds, float errorBoundAvg) {}
