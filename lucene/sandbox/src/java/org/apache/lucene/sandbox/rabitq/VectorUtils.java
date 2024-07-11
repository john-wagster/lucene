package org.apache.lucene.sandbox.rabitq;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import static jdk.incubator.vector.VectorOperators.ADD;

// FIXME: FUTURE - copied from lucene internals for now
public class VectorUtils {

    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;;

    public static float squareDistance(float[] a, float[] b) {
        return squareDistance(a, b, a.length);
    }

    private static float squareDistance(float[] a, float[] b, int limit) {
        int i = 0;
        // vector loop is unrolled 4x (4 accumulators in parallel)
        // we don't know how many the cpu can do at once, some can do 2, some 4
        FloatVector acc1 = FloatVector.zero(FLOAT_SPECIES);
        FloatVector acc2 = FloatVector.zero(FLOAT_SPECIES);
        FloatVector acc3 = FloatVector.zero(FLOAT_SPECIES);
        FloatVector acc4 = FloatVector.zero(FLOAT_SPECIES);
        int unrolledLimit = limit - 3 * FLOAT_SPECIES.length();
        for (; i < unrolledLimit; i += 4 * FLOAT_SPECIES.length()) {
            // one
            FloatVector va = FloatVector.fromArray(FLOAT_SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(FLOAT_SPECIES, b, i);
            FloatVector diff1 = va.sub(vb);
            acc1 = fma(diff1, diff1, acc1);

            // two
            FloatVector vc = FloatVector.fromArray(FLOAT_SPECIES, a, i + FLOAT_SPECIES.length());
            FloatVector vd = FloatVector.fromArray(FLOAT_SPECIES, b, i + FLOAT_SPECIES.length());
            FloatVector diff2 = vc.sub(vd);
            acc2 = fma(diff2, diff2, acc2);

            // three
            FloatVector ve = FloatVector.fromArray(FLOAT_SPECIES, a, i + 2 * FLOAT_SPECIES.length());
            FloatVector vf = FloatVector.fromArray(FLOAT_SPECIES, b, i + 2 * FLOAT_SPECIES.length());
            FloatVector diff3 = ve.sub(vf);
            acc3 = fma(diff3, diff3, acc3);

            // four
            FloatVector vg = FloatVector.fromArray(FLOAT_SPECIES, a, i + 3 * FLOAT_SPECIES.length());
            FloatVector vh = FloatVector.fromArray(FLOAT_SPECIES, b, i + 3 * FLOAT_SPECIES.length());
            FloatVector diff4 = vg.sub(vh);
            acc4 = fma(diff4, diff4, acc4);
        }
        // vector tail: less scalar computations for unaligned sizes, esp with big vector sizes
        for (; i < limit; i += FLOAT_SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(FLOAT_SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(FLOAT_SPECIES, b, i);
            FloatVector diff = va.sub(vb);
            acc1 = fma(diff, diff, acc1);
        }
        // reduce
        FloatVector res1 = acc1.add(acc2);
        FloatVector res2 = acc3.add(acc4);
        return res1.add(res2).reduceLanes(ADD);
    }

    private static FloatVector fma(FloatVector a, FloatVector b, FloatVector c) {
        return a.fma(b, c);
    }
}
