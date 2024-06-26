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
    public static void main(String[] args) throws Exception {
        // FIXME: FUTURE - allow the output path to be settable
        // index DIRECTORY_TO_DATASET DATASET_NAME NUM_CENTROIDS DIMENSIONS
        String source = args[0];
        String dataset = args[1];
        int kTotalClusters = Integer.parseInt(args[2]);
        int dimensions = Integer.parseInt(args[3]);

        // FIXME: FUTURE - switch these to log statements
        System.out.println("Clustering - " + dataset);
        clusterWithIVF(source, dataset, kTotalClusters);

        // FIXME: FUTURE - switch these to log statements
        System.out.println("Generating subspaces - " + dataset);
        generateSubSpaces(source, dataset, kTotalClusters);

        int C = kTotalClusters;
        int D = dimensions;
        int B = (D + 63) / 64 * 64;

        // FIXME: FUTURE - clean up this gross path mgmt
        String dataPath = String.format("%s%s_base.fvecs", source, dataset);
        float[][] X = IOUtils.readFvecs(new FileInputStream(dataPath));

        String centroidPath = String.format("%sRandCentroid_C%d_B%d.fvecs", source, C, B);
        float[][] centroids = IOUtils.readFvecs(new FileInputStream(centroidPath));

        String x0Path = String.format("%sx0_C%d_B%d.fvecs", source, C, B);
        float[] x0 = IOUtils.readFvecsCalcs(new FileInputStream(x0Path));

        String distToCentroidPath = String.format("%s%s_dist_to_centroid_%d.fvecs", source, dataset, C);
        float[] distToCentroid = IOUtils.readFvecsCalcs(new FileInputStream(distToCentroidPath));

        String clusterIdPath = String.format("%s%s_cluster_id_%d.ivecs", source, dataset, C);
        int[] clusterId = IOUtils.readIvecsCalcs(new FileInputStream(clusterIdPath));

        String binaryPath = String.format("%sRandNet_C%d_B%d.ivecs", source, C, B);
        long[][] binary = IOUtils.readBinary(new FileInputStream(binaryPath));

        String indexPath = String.format("%sivfrabitq%d_B%d.index", source, C, B);
        System.out.println("Loading Succeed!");
        IVFRN ivf = new IVFRN(X, centroids, distToCentroid, x0, clusterId, binary);

        // Save index
        ivf.save(indexPath);
    }

    public static void clusterWithIVF(String source, String dataset, int kTotalClusters) throws IOException {
        Path basePath = Paths.get(source);
        Path fvecPath = Paths.get(basePath.toString(), dataset + "_base.fvecs");

        float[][] X = IOUtils.readFvecs(new FileInputStream(fvecPath.toFile())); // X
        Path centroidsPath = Paths.get(basePath.toString(), dataset + "_centroid_" + kTotalClusters + ".fvecs");
        Path distToCentroidPath = Paths.get(basePath.toString(), dataset + "_dist_to_centroid_" + kTotalClusters + ".fvecs");
        Path clusterIdPath = Paths.get(basePath.toString(), dataset + "_cluster_id_" + kTotalClusters + ".ivecs");

        // cluster data vectors
        System.out.println("cluster data vectors");
        IVF index = new IVF(kTotalClusters);
        index.train(X);
        Centroid[] centroids = index.getCentroids();
        SearchResult[] results = index.search(X);
        float[] distToCentroids = new float[results.length];
        int[] clusterIds = new int[results.length];
        for(int i = 0; i < results.length; i++) {
            SearchResult result = results[i];
            float distToCentroid = result.distToCentroid();
            clusterIds[i] = result.clusterId();
            distToCentroids[i] = (float) Math.pow(distToCentroid, 0.5);
        }

        IOUtils.toFvecsCalcs(new FileOutputStream(distToCentroidPath.toFile()), distToCentroids);
        IOUtils.toIvecsCalcs(new FileOutputStream(clusterIdPath.toFile()), clusterIds);
        IOUtils.toFvecs(new FileOutputStream(centroidsPath.toFile()), Arrays.stream(centroids).map(Centroid::getVector).toArray(float[][]::new));
    }

    public static void generateSubSpaces(String source, String dataset, int kTotalClusters) throws IOException {
        // FIXME: FUTURE - clean this usage up
//        String[] datasets = {dataset};
        String path = Paths.get(source).toString();

        // FIXME: FUTURE - support processing multiple datasets
//        for (String dataset : datasets) {
        String dataPath = new File(path, dataset + "_base.fvecs").getAbsolutePath();

        int C = kTotalClusters;
        String centroidsPath = new File(path, dataset + "_centroid_" + C + ".fvecs").getAbsolutePath();
        String clusterIdPath = new File(path, dataset + "_cluster_id_" + C + ".ivecs").getAbsolutePath();

        float[][] X = IOUtils.readFvecs(new FileInputStream(dataPath));
        float[][] centroids = IOUtils.readFvecs(new FileInputStream(centroidsPath));
        int[] clusterId = IOUtils.readIvecsCalcs(new FileInputStream(clusterIdPath));

        int D = X[0].length;
        int B = (D + 63) / 64 * 64;
        int MAX_BD = Math.max(D, B);

        Path projectionPath = Paths.get(new File(path, "P_C" + C + "_B" + B + ".fvecs").getAbsolutePath());
        Path randomizedCentroidPath = Paths.get(new File(path, "RandCentroid_C" + C + "_B" + B + ".fvecs").getAbsolutePath());
        Path RNPath = Paths.get(new File(path, "RandNet_C" + C + "_B" + B + ".ivecs").getAbsolutePath());
        Path x0Path = Paths.get(new File(path, "x0_C" + C + "_B" + B + ".fvecs").getAbsolutePath());

        float[][] P = getOrthogonalMatrix(MAX_BD);

        // The inverse of an orthogonal matrix equals to its transpose.
        P = MatrixUtils.transpose(P);

        float[][] XPad = MatrixUtils.padColumns(X, MAX_BD-D);
        float[][] centroidsPad = MatrixUtils.padColumns(centroids, MAX_BD-D);

        float[][] XP = MatrixUtils.dotProduct(XPad, P);
        float[][] CP = MatrixUtils.dotProduct(centroidsPad, P);
        XP = MatrixUtils.subtract(XP, MatrixUtils.subset(CP, clusterId));

        boolean[][] binXP = MatrixUtils.greaterThan(XP, 0);

        // The inner product between the data vector and the quantized data vector
        float[][] XPSubset = MatrixUtils.subset(XP, B);
        boolean[][] binXPSubset = MatrixUtils.subset(binXP, B);
        float[][] binXPSubsetAsInts = MatrixUtils.asFloats(binXPSubset);
        float[][] XPSubsetDotbinXP = MatrixUtils.multiplyElementWise(XPSubset, binXPSubsetAsInts);
        float[][] XPSubsetDotBinXpNormalized = MatrixUtils.divide(XPSubsetDotbinXP, (float) Math.pow(B, 0.5));
        float[][] XPSubsetSummedRows = MatrixUtils.sumRows(XPSubsetDotBinXpNormalized);
        float[][] x0 = MatrixUtils.normalize(XPSubsetSummedRows, MatrixUtils.normsForRows(XP));
        x0 = MatrixUtils.replaceInfinite(x0, 0.8f);

        long[][] repackedBinXP = MatrixUtils.repackAsUInt64(binXP, B);

        IOUtils.toFvecs(new FileOutputStream(randomizedCentroidPath.toFile()), CP);
        IOUtils.toIvecs(new FileOutputStream(RNPath.toFile()), repackedBinXP);
        IOUtils.toFvecsCalcs(new FileOutputStream(x0Path.toFile()), MatrixUtils.flatten(x0));
        IOUtils.toFvecs(new FileOutputStream(projectionPath.toFile()), P);
//        }
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
        return qr.getQ();
    }
}

