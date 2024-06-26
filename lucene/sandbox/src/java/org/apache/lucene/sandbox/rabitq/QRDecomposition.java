package org.apache.lucene.sandbox.rabitq;

// FIXME: FUTURE - evaluate pulling in apache commons Math (this is a rough pull from that codebase; apache licensed)
class QRDecomposition {
    private float[][] qrt;
    private float[] rDiag;
    private float[][] cachedQ;
    private float[][] cachedQT;

    public QRDecomposition(float[][] matrix) {
        final int m = matrix.length;
        final int n = matrix[0].length;
        qrt = MatrixUtils.transpose(matrix);
        rDiag = new float[Math.min(m, n)];
        cachedQ = null;
        cachedQT = null;

        decompose(qrt);
    }

    protected void decompose(float[][] matrix) {
        for (int minor = 0; minor < Math.min(matrix.length, matrix[0].length); minor++) {
            performHouseholderReflection(minor, matrix);
        }
    }

    protected void performHouseholderReflection(int minor, float[][] matrix) {

        final float[] qrtMinor = matrix[minor];

        double xNormSqr = 0;
        for (int row = minor; row < qrtMinor.length; row++) {
            final double c = qrtMinor[row];
            xNormSqr += c * c;
        }
        final float a = (float) ((qrtMinor[minor] > 0) ? -Math.sqrt(xNormSqr) : Math.sqrt(xNormSqr));
        rDiag[minor] = a;

        if (a != 0.0) {
            qrtMinor[minor] -= a; // now |v|^2 = -2a*(qr[minor][minor])

            for (int col = minor + 1; col < matrix.length; col++) {
                final float[] qrtCol = matrix[col];
                double alpha = 0;
                for (int row = minor; row < qrtCol.length; row++) {
                    alpha -= qrtCol[row] * qrtMinor[row];
                }
                alpha /= a * qrtMinor[minor];

                // Subtract the column vector alpha*v from x.
                for (int row = minor; row < qrtCol.length; row++) {
                    qrtCol[row] -= alpha * qrtMinor[row];
                }
            }
        }
    }

    public float[][] getQ() {
        if (cachedQ == null) {
            cachedQ = MatrixUtils.transpose(getQT());
        }
        return cachedQ;
    }

    public float[][] getQT() {
        if (cachedQT == null) {
            final int n = qrt.length;
            final int m = qrt[0].length;
            float[][] qta = new float[m][m];

            for (int minor = m - 1; minor >= Math.min(m, n); minor--) {
                qta[minor][minor] = 1.0f;
            }

            for (int minor = Math.min(m, n) - 1; minor >= 0; minor--) {
                final float[] qrtMinor = qrt[minor];
                qta[minor][minor] = 1.0f;
                if (qrtMinor[minor] != 0.0) {
                    for (int col = minor; col < m; col++) {
                        double alpha = 0;
                        for (int row = minor; row < m; row++) {
                            alpha -= qta[col][row] * qrtMinor[row];
                        }
                        alpha /= rDiag[minor] * qrtMinor[minor];

                        for (int row = minor; row < m; row++) {
                            qta[col][row] += -alpha * qrtMinor[row];
                        }
                    }
                }
            }
            cachedQT = qta;
        }

        return cachedQT;
    }
}
