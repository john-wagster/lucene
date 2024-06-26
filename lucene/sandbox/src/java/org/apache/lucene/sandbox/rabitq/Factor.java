package org.apache.lucene.sandbox.rabitq;

record Factor(
    float sqrX,
    float error,
    float factorPPC,
    float factorIP
){}
