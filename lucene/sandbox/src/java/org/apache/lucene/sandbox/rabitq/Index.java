package org.apache.lucene.sandbox.rabitq;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;


public class Index {
    // FIXME: introduce logging instead of println
    // FIXME: stream in X so it can be loaded into memory
    // FIXME: better arg parsing

    public static void main(String[] args) throws Exception {
        String source = args[0];
        String dataset = args[1];
        int numCentroids = Integer.parseInt(args[2]);

        Path basePath = Paths.get(source);
        Path fvecPath = Paths.get(basePath.toString(), dataset + "_base.fvecs");

        float[][] X = IOUtils.readFvecs(new FileInputStream(fvecPath.toFile())); // X
        int D = X[0].length;
        int B = (D + 63) / 64 * 64;

        System.out.println("Clustering - " + dataset);
        long startTime = System.nanoTime();
        IVFOutput ivfOutput = clusterWithIVF(X, numCentroids);
        System.out.println("Time to compute IVF: " + (System.nanoTime() - startTime) / 1e9);

        System.out.println("Generating subspaces - " + dataset);
        startTime = System.nanoTime();
        int MAX_BD = Math.max(D, B);

        Path projectionPath = Paths.get(new File(source, "P_C" + numCentroids + "_B" + B + ".fvecs").getAbsolutePath());
        float[][] P = getOrthogonalMatrix(MAX_BD);
        MatrixUtils.transpose(P);
        IOUtils.toFvecs(new FileOutputStream(projectionPath.toFile()), P);

        SubspaceOutput subspaceOutput = generateSubSpaces(D, B, MAX_BD, P, X, ivfOutput.centroidVectors(), ivfOutput.clusterIds());
        System.out.println("Time to compute sub-spaces: " + (System.nanoTime() - startTime) / 1e9);

        float[][] centroids = subspaceOutput.cp();
        float[] x0 = subspaceOutput.x0();
        float[] distToCentroid = ivfOutput.distToCentroids();
        int[] clusterId = ivfOutput.clusterIds();
        long[][] binary = subspaceOutput.repackedBinXP();

        System.out.println("Loading Complete!");
        IVFRN ivf = new IVFRN(X, centroids, distToCentroid, x0, clusterId, binary);

        System.out.println("Saving");
        String indexPath = source + "ivfrabitq" + numCentroids + "_B" + B + ".index";
        ivf.save(indexPath);
    }

    private static IVFOutput clusterWithIVF(float[][] X, int numCentroids) throws IOException {

        // cluster data vectors
        System.out.println("cluster data vectors");
        SamplingIVF index = new SamplingIVF(numCentroids);
        index.train(X);
        Centroid[] centroids = index.getCentroids();
        SearchResult[] results = index.search(X);
        float[] distToCentroids = new float[results.length];
        int[] clusterIds = new int[results.length];
        for (int i = 0; i < results.length; i++) {
            SearchResult result = results[i];
            float distToCentroid = result.getDistToCentroid();
            clusterIds[i] = result.getClusterId();
            distToCentroids[i] = (float) Math.pow(distToCentroid, 0.5);
        }

        float[][] centroidVectors = Arrays.stream(centroids).map(Centroid::getVector).toArray(float[][]::new);

        return new IVFOutput(distToCentroids, clusterIds, centroidVectors);
    }

    private static SubspaceOutput generateSubSpaces(int D, int B, int MAX_BD, float[][] P, float[][] X, float[][] centroids, int[] clusterIds) throws IOException {
        float[][] XP = MatrixUtils.padColumns(X, MAX_BD-D); // typically no-op if D/64
        float[][] CP = MatrixUtils.padColumns(centroids, MAX_BD-D); // typically no-op if D/64

        XP = MatrixUtils.dotProduct(XP, P);
        CP = MatrixUtils.dotProduct(CP, P);
        XP = MatrixUtils.subtract(XP, CP, clusterIds);

        // The inner product between the data vector and the quantized data vector
        float[][] XPSubset = MatrixUtils.subset(XP, B); // typically no-op if D/64
        MatrixUtils.removeSignAndDivide(XPSubset, (float) Math.pow(B, 0.5));
        float[] x0 = MatrixUtils.sumAndNormalize(XPSubset, MatrixUtils.normsForRows(XP));

        long[][] repackedBinXP = MatrixUtils.repackAsUInt64(XP, B);

        return new SubspaceOutput(P, CP, x0, repackedBinXP);
    }

    private static float[][] getOrthogonalMatrix(int d) {
        Random random = new Random();
        float[][] G = new float[d][d];
        for(int i = 0; i < d; i++) {
            for(int j = 0; j < d; j++) {
                G[i][j] = (float) random.nextGaussian();
            }
        }

        QRDecomposition qr = new QRDecomposition(G);
        return qr.getQT();
    }
}

