package org.apache.lucene.sandbox.rabitq;

import java.util.PriorityQueue;

public record IVFRNResult(PriorityQueue<Result> results, IVFRNStats stats) {}
