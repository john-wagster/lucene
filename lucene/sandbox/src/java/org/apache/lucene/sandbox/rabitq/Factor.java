package org.apache.lucene.sandbox.rabitq;

public record Factor(
    float sqrX,
    float error,
    float factorPPC,
    float factorIP
){}
