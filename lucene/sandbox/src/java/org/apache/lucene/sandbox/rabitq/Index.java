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
        int dimensions = Integer.parseInt(args[3]);

        Path basePath = Paths.get(source);
        Path fvecPath = Paths.get(basePath.toString(), dataset + "_base.fvecs");

        int D = dimensions;
        int B = (D + 63) / 64 * 64;

        System.out.println("Clustering - " + dataset);
        long startTime = System.nanoTime();
        IVFOutput ivfOutput = clusterWithIVF(fvecPath, numCentroids, dimensions);
        System.out.println("Time to compute IVF: " + (System.nanoTime() - startTime) / 1e9);

        System.out.println("Generating subspaces - " + dataset);
        startTime = System.nanoTime();
        int MAX_BD = Math.max(D, B);

        Path projectionPath = Paths.get(new File(source, "P_C" + numCentroids + "_B" + B + ".fvecs").getAbsolutePath());
        float[][] P = getOrthogonalMatrix(MAX_BD);
        MatrixUtils.transpose(P);
        IOUtils.toFvecs(new FileOutputStream(projectionPath.toFile()), P);

        SubspaceOutput subspaceOutput = generateSubSpaces(D, B, MAX_BD, P, fvecPath, ivfOutput.centroidVectors(), ivfOutput.clusterIds());

        System.out.println("Time to compute sub-spaces: " + (System.nanoTime() - startTime) / 1e9);

        float[][] centroids = subspaceOutput.cp();
        float[] x0 = subspaceOutput.x0();
        float[] distToCentroid = ivfOutput.distToCentroids();
        int[] clusterId = ivfOutput.clusterIds();
        long[][] binary = subspaceOutput.repackedBinXP();

        System.out.println("Loading Complete!");
        IVFRN ivf = new IVFRN(fvecPath, centroids, distToCentroid, x0, clusterId, binary, dimensions);

        System.out.println("Saving");
        String indexPath = source + "ivfrabitq" + numCentroids + "_B" + B + ".index";
        ivf.save(indexPath);
    }

    private static IVFOutput clusterWithIVF(Path XPath, int numCentroids, int dimensions) throws IOException {

        // cluster data vectors
        System.out.println("cluster data vectors");
        SamplingIVF index = new SamplingIVF(numCentroids);
        index.train(XPath, dimensions);
        Centroid[] centroids = index.getCentroids();

        SearchResult[] results;
        try(FvecsStream XStream = IOUtils.createFvecsStream(new FileInputStream(XPath.toFile()), dimensions)) { // X
            results = index.search(XStream);
        }

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

    private static SubspaceOutput generateSubSpaces(int D, int B, int MAX_BD, float[][] P, Path XPath, float[][] centroids, int[] clusterIds) throws IOException {
        try(FvecsStream XStream = IOUtils.createFvecsStream(new FileInputStream(XPath.toFile()), D)) { // X

            int XLength = XStream.getTotalFvecs();

            float[][] CP = MatrixUtils.padColumns(centroids, MAX_BD - D); // typically no-op if D/64
            float[] x0 = new float[XLength];
            long[][] repackedBinXP = new long[XLength][B >> 6];

            CP = MatrixUtils.dotProduct(CP, P);

            for (int i = 0; i < XLength; i++) {
                float[] X = XStream.getNextFvec();
                float[] XP = MatrixUtils.partialPadColumns(X, MAX_BD - D); // typically no-op if D/64

                XP = MatrixUtils.partialDotProduct(XP, P);
                XP = MatrixUtils.partialSubtract(XP, CP[clusterIds[i]]);

                // The inner product between the data vector and the quantized data vector
                float norm = MatrixUtils.partialNormForRow(XP);
                float[] XPSubset = MatrixUtils.partialSubset(XP, B); // typically no-op if D/64
                MatrixUtils.partialRemoveSignAndDivide(XPSubset, (float) Math.pow(B, 0.5));
                x0[i] = MatrixUtils.partialSumAndNormalize(XPSubset, norm);

                repackedBinXP[i] = MatrixUtils.partialRepackAsUInt64(XP, B);
            }
            return new SubspaceOutput(P, CP, x0, repackedBinXP);
        }
    }

    private static float[][] getOrthogonalMatrix(int d) {
        Random random = new Random(1);
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

