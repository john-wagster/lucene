package org.apache.lucene.sandbox.rabitq;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;

public class MatrixUtils {
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;;

    public static void removeSignAndDivide(float[][] a, float divisor) {
        for(int i = 0; i < a.length; i++) {
            // FIXME: revert to old behavior for small dimensions
//            for(int j = 0; j < a[0].length; j++) {
//                a[i][j] = Math.abs(a[i][j]) / divisor;
//            }
            int size = a[0].length / FLOAT_SPECIES.length();
            for(int r = 0; r < size; r++) {
                int offset = FLOAT_SPECIES.length() * r;
                FloatVector va = FloatVector.fromArray(FLOAT_SPECIES, a[i], offset);
                va.abs().div(divisor).intoArray(a[i], offset);
            }
        }
    }

    public static float[] sumAndNormalize(float[][] a, float[] norms) {
        // FIXME: FUTURE - throw errors here for norms being the incorrect or unexpected shape
        float[] aDivided = new float[a.length];
        for(int i = 0; i < a.length; i++) {
            for(int j = 0; j < a[0].length; j++) {
                aDivided[i] += a[i][j];
            }
            aDivided[i] = aDivided[i] / norms[i];
        }

        return aDivided;
    }

    public static float distance(float[][] a, int startA, float[][] b, int startB) {
        float[] vectorA = a[startA];
        float[] vectorB = b[startB];
        return VectorUtils.squareDistance(vectorA, vectorB);
    }

    public static void transpose(float[][] a) {
        int m = a.length;
        int n = a[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = i+1; j < n; j++) {
                if( i != j) {
                    float tmp = a[i][j];
                    a[i][j] = a[j][i];
                    a[j][i] = tmp;
                }
            }
        }
    }

    public static float[][] dotProduct(float[][] a, float[][] b) {
        int m = a.length;
        int n = a[0].length;
        int bN = b[0].length;
        if (n != b.length) {
            throw new IllegalArgumentException("Matrices are not compatible for dot product");
        }
        float[][] result = new float[m][bN];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < bN; j++) {
                for (int k = 0; k < n; k++) {
                    result[i][j] = Math.fma(a[i][k], b[k][j], result[i][j]);
                }
            }
        }
        return result;
    }

    public static float[][] subset(float[][] a, int lastColumn) {
        if(a.length == lastColumn) {
            return a;
        }
        // FIXME: FUTURE - check error conditions
        float[][] subMatrix = new float[a.length][lastColumn];
        for(int i = 0; i < a.length; i++) {
            subMatrix[i] = Arrays.copyOf(a[i], lastColumn);
        }
        return subMatrix;
    }

    public static float[][] subtract(float[][] a, float[][] b, int[] indicies) {
        // FIXME: FUTURE - check error conditions
        float[][] result = new float[a.length][a[0].length];
        for(int i = 0; i < a.length; i++) {
            float[] c = b[indicies[i]];
            for(int j = 0; j < a[i].length; j++) {
                result[i][j] = a[i][j] - c[j];
            }
        }

        return result;
    }

    public static boolean[][] greaterThan(float[][] a, float v) {
        // FIXME: FUTURE - check error conditions
        boolean[][] conditions = new boolean[a.length][a[0].length];
        for(int i = 0; i < a.length; i++) {
            for(int j = 0; j < a[0].length; j++) {
                conditions[i][j] = a[i][j] > v;
            }
        }

        return conditions;
    }

    public static float[][] padColumns(float[][] a, int dimension) {
        if(dimension == 0) {
            return a;
        }
        int pad = Math.max(a[0].length, dimension);
        float[][] aPad = new float[a.length][pad];
        for(int i = 0; i < a.length; i++) {
            for(int j = 0; j < pad; j++) {
                if( j < a[0].length ) {
                    aPad[i][j] = a[i][j];
                } else {
                    aPad[i][j] = 0;
                }
            }
        }
        return aPad;
    }

    public static float[] normsForRows(float[][] m) {
        float[] normalized = new float[m.length];
        for(int h = 0; h < m.length; h++) {
            float[] vector = m[h];
            // Calculate magnitude/length of the vector
            double magnitude = 0;
            for (int i = 0; i < vector.length; i++) {
                magnitude = Math.fma(vector[i], vector[i], magnitude);
            }
            magnitude = Math.sqrt(magnitude);

            // FIXME: FUTURE - not good; sometimes this needs to be 0
//            if (magnitude == 0) {
//                throw new IllegalArgumentException("Cannot normalize a vector of length zero.");
//            }

            normalized[h] = (float) magnitude;
        }

        return normalized;
    }

    public static void replaceInfinite(float[] a, float value) {
        // FIXME: FUTURE - handle errors
        for(int i = 0; i < a.length; i++) {
            if (!Float.isFinite(a[i])) {
                a[i] = value;
            }
        }
    }

    public static boolean[] flatten(boolean[][] a) {
        boolean[] aFlattened = new boolean[a.length * a[0].length];
        for(int i = 0; i < a.length; i++) {
            for(int j = 0; j < a[0].length; j++) {
                aFlattened[i*(a[0].length)+j] = a[i][j];
            }
        }
        return aFlattened;
    }

    public static long[][] repackAsUInt64(boolean[][] binXP, int B) {
        boolean[] binXPFlattened = MatrixUtils.flatten(binXP);

        int totalValues = B >> 6;
        int vectorSize = (binXPFlattened.length / 64) / totalValues;

        long[] binaryData = new long[vectorSize * totalValues];

        int idx = 0;
        for(int h = 0; h < binXPFlattened.length / 64; h++) {
            long result = 0L;
            int q = 0;
            for (int i = 7; i >= 0; i--) {
                for (int j = 7; j >= 0; j--) {
                    if (binXPFlattened[h * 64 + i * 8 + j]) {
                        result |= (1L << q);
                    }
                    q++;
                }
            }

            binaryData[idx] = result;
            idx++;
        }

        long[][] allBinary = new long[vectorSize][totalValues];

        idx = 0;
        for(int i = 0; i < vectorSize; i++) {
            for(int j = 0; j < totalValues; j++) {
                allBinary[i][j] = binaryData[idx];
                idx++;
            }
        }

        return allBinary;
    }
}
