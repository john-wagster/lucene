package org.apache.lucene.sandbox.rabitq;

import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.TimeUnit;


public class Index {
    // FIXME: introduce logging instead of println
    // FIXME: stream in X so it can be loaded into memory
    // FIXME: better arg parsing
    static final int QUORA_E5_DOC_SIZE = 522_931;
    static final int NUM_DOCS = QUORA_E5_DOC_SIZE;
    // The number of docs to sample to compute the coarse clustering.
    static final int COARSE_CLUSTERING_SAMPLE_SIZE = (int)(QUORA_E5_DOC_SIZE * 0.1);
    // The number of iterations to run k-means for when computing the coarse clustering.
    static final int COARSE_CLUSTERING_KMEANS_ITR = 10; // 10;
    // The number of random restarts of clustering to use when computing the coarse clustering.
    static final int COARSE_CLUSTERING_KMEANS_RESTARTS = 5; // 5;

    public static void main(String[] args) throws Exception {
        String source = "/Users/benjamintrent/rabit_data/"; //args[0];
        int numCentroids = 1;// Integer.parseInt(args[2]);
        int dimensions = 384;//Integer.parseInt(args[3]);
        Path basePath = Paths.get(source);
        int D = dimensions;
        int B = (D + 63) / 64 * 64;
        float[][] P;
        try (MMapDirectory directory = new MMapDirectory(basePath);
             IndexInput vectorInput = directory.openInput(source + "quora-522k-e5small_corpus-quora-E5-small.fvec", IOContext.DEFAULT)) {
            RandomAccessVectorValues.Floats vectorValues =
              new VectorsReaderWithOffset(vectorInput, NUM_DOCS, dimensions, dimensions * Float.BYTES, Float.BYTES);
            System.out.println("Clustering - e5small");
            long startTime = System.nanoTime();
            IVFOutput ivfOutput = clusterWithIVF(vectorValues, numCentroids, dimensions);
            long nanosToComputeIVF = System.nanoTime() - startTime;
            System.out.println("Time to compute IVF: " + TimeUnit.NANOSECONDS.toMillis(nanosToComputeIVF));
            System.out.println("Generating subspaces - e5small");
            startTime = System.nanoTime();
            int MAX_BD = Math.max(D, B);
            P = getOrthogonalMatrix(MAX_BD);
            MatrixUtils.transpose(P);
            SubspaceOutput subspaceOutput = generateSubSpaces(D, B, MAX_BD, P, vectorValues, ivfOutput.centroidVectors(), ivfOutput.clusterIds());
            System.out.println("Time to compute sub-spaces: " + TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startTime));
            Path projectionPath = Paths.get(new File(source, "P_C" + numCentroids + "_B" + B + ".fvecs").getAbsolutePath());
            IOUtils.toFvecs(new FileOutputStream(projectionPath.toFile()), P);
            float[][] centroids = subspaceOutput.cp();
            float[] x0 = subspaceOutput.x0();
            float[] distToCentroid = ivfOutput.distToCentroids();
            int[] clusterId = ivfOutput.clusterIds();
            long[][] binary = subspaceOutput.repackedBinXP();

            System.out.println("Loading Complete!");
            IVFRN ivf = new IVFRN(NUM_DOCS, centroids, distToCentroid, x0, clusterId, binary, dimensions);

            System.out.println("Saving");
            String indexPath = source + "ivfrabitq" + numCentroids + "_B" + B + ".index";
            ivf.save(indexPath);
        }
    }

    private static IVFOutput clusterWithIVF(RandomAccessVectorValues.Floats vectors, int numCentroids, int dimensions) throws IOException {

        // cluster data vectors
        System.out.println("cluster data vectors");
        SamplingIVF index = new SamplingIVF(numCentroids);
        index.train(vectors, dimensions);
        Centroid[] centroids = index.getCentroids();

        SearchResult[] results = index.search(vectors);
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

    private static SubspaceOutput generateSubSpaces(int D, int B, int MAX_BD, float[][] P, RandomAccessVectorValues.Floats vectorValues, float[][] centroids, int[] clusterIds) throws IOException {
         int XLength = vectorValues.size();

         float[][] CP = MatrixUtils.padColumns(centroids, MAX_BD - D); // typically no-op if D/64
        // TODO This is  bad, loading all vectors into memory is untenable. We will need to transform this.
         float[] x0 = new float[XLength];
         long[][] repackedBinXP = new long[XLength][B >> 6];

         CP = MatrixUtils.dotProduct(CP, P);
         for (int i = 0; i < vectorValues.size(); i++) {
             float[] X = vectorValues.vectorValue(i);
             float[] XP = MatrixUtils.partialPadColumns(X, MAX_BD - D); // typically no-op if D/64

             XP = MatrixUtils.partialDotProduct(XP, P);
             MatrixUtils.partialSubtract(XP, CP[clusterIds[i]]);

             // The inner product between the data vector and the quantized data vector
             float norm = MatrixUtils.partialNormForRow(XP);
             float[] XPSubset = MatrixUtils.partialSubset(XP, B); // typically no-op if D/64
             MatrixUtils.partialRemoveSignAndDivide(XPSubset, (float) Math.pow(B, 0.5));
             x0[i] = MatrixUtils.partialSumAndNormalize(XPSubset, norm);
             repackedBinXP[i] = MatrixUtils.partialRepackAsUInt64(XP, B);
         }
         return new SubspaceOutput(P, CP, x0, repackedBinXP);
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

