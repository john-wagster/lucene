package org.apache.lucene.sandbox.rabitq;

public record IVFRNStats(int totalExploredNNs, int totalComparisons, int maxEstimatorSize, int totalEstimatorQueueAdds, float errorBoundAvg) {}
