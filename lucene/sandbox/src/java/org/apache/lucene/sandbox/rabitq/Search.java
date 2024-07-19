package org.apache.lucene.sandbox.rabitq;

import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.store.ReadAdvice;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.PriorityQueue;

public class Search {

    public static void main(String[] args) throws Exception {
        // FIXME: better arg parsing
        // FIXME: clean up gross path mgmt
        // search DIRECTORY_TO_DATASET DATASET_NAME NUM_CENTROIDS DIMENSIONS B_QUERY OUTPUT_PATH
        String source = "/Users/benjamintrent/rabit_data/";  // eg "/Users/jwagster/Desktop/gist1m/gist/"
        int numCentroids = 1;//Integer.parseInt(args[2]);
        int dimensions = 384;//Integer.parseInt(args[3]);
        int B_QUERY = 4;//Integer.parseInt(args[4]);
        int k = 100;//Integer.parseInt(args[5]);
        int B = (dimensions + 63) / 64 * 64;
        Path basePath = Paths.get(source);

        String queryPath = String.format("%s%s", source, "quora-522k-e5small_queries-quora-E5-small.fvec");
        String dataPath = String.format("%s%s", source, "quora-522k-e5small_corpus-quora-E5-small.fvec");
        String groundTruthPath = String.format("%s%s", source, "quora-522k-e5small_groundtruth-quora-E5-small.ivec");
        int[][] G = IOUtils.readIvecs(new FileInputStream(groundTruthPath));
        String transformationPath = String.format("%sP_C%d_B%d.fvecs", source, numCentroids, B);
        float[][] P = IOUtils.readFvecs(new FileInputStream(transformationPath));
        String indexPath = String.format("%sivfrabitq%d_B%d.index", source, numCentroids, B);
        IVFRN ivfrn = IVFRN.load(indexPath);
        try (MMapDirectory directory = new MMapDirectory(basePath);
             IndexInput vectorInput =
               directory.openInput(dataPath, IOContext.DEFAULT.withReadAdvice(ReadAdvice.RANDOM));
             IndexInput queryInput = directory.openInput(queryPath, IOContext.READONCE)) {
            RandomAccessVectorValues.Floats queryVectors = new VectorsReaderWithOffset(queryInput, 1000, dimensions);
            float[][] Q = new float[queryVectors.size()][dimensions];
            for (int i = 0; i < queryVectors.size(); i++) {
                Q[i] = Arrays.copyOf(queryVectors.vectorValue(i), dimensions);
            }

            RandomAccessVectorValues.Floats dataVectors = new VectorsReaderWithOffset(vectorInput, Index.QUORA_E5_DOC_SIZE, dimensions);
            // we got to stop this, projecting should be measured as part of the query time. This is cheating slightly
            float[][] RandQ = MatrixUtils.dotProduct(Q, P);

            test(queryVectors, RandQ, dataVectors, G, ivfrn, k, B_QUERY);
        }
    }

    public static void test(RandomAccessVectorValues.Floats queryVectors, float[][] RandQ, RandomAccessVectorValues.Floats dataVectors, int[][] G, IVFRN ivf, int k, int B_QUERY) throws IOException {

        int nprobes = 300;
        nprobes = Math.min(nprobes, ivf.getC()); // FIXME: hardcoded
        assert nprobes <= k;

        float totalUsertime = 0;
        float totalRatio = 0;
        int correctCount = 0;

        float errorBoundAvg = 0f;
        int maxEstimatorSize = 0;
        int totalEstimatorAdds = 0;
        System.out.println("Starting search");
        for (int i = 0; i < queryVectors.size(); i++) {
            long startTime = System.nanoTime();
            float[] queryVector = queryVectors.vectorValue(i);
            float[] randQ = RandQ[i];
            IVFRNResult result = ivf.search(dataVectors, queryVector, randQ, k, nprobes, B_QUERY);
            PriorityQueue<Result> KNNs = result.results();
            IVFRNStats stats = result.stats();
            float usertime = (System.nanoTime() - startTime) / 1e3f; // convert to microseconds to compare to the c impl
            totalUsertime += usertime;

            PriorityQueue<Result> copyOfKNN = new PriorityQueue<>();
            copyOfKNN.addAll(KNNs);

            int correct = 0;
            while (!KNNs.isEmpty()) {
                int id = KNNs.remove().c();
                for (int j = 0; j < k; j++) {
                    if (id == G[i][j]) {
                        correct++;
                    }
                }
            }
            correctCount += correct;

            if (i % 1500 == 0) {
                System.out.print(".");
            }

            errorBoundAvg += stats.errorBoundAvg();
            maxEstimatorSize = stats.maxEstimatorSize();
            totalEstimatorAdds += stats.totalEstimatorQueueAdds();
        }
        System.out.println();

        // FIXME: missing rotation time?
        float timeUsPerQuery = totalUsertime / queryVectors.size();
        float recall = (float) correctCount / (queryVectors.size() * k);
        float averageRatio = totalRatio / (queryVectors.size() * k);

        System.out.println("------------------------------------------------");
        System.out.println("nprobe = " + nprobes + "\tk = " + k + "\tCoarse Clusters = " + ivf.getC());
        System.out.println("Recall = " + recall * 100f + "%\t" + "Ratio = " + averageRatio);
        System.out.println("Avg Time Per Search = " + timeUsPerQuery + " us\t QPS = " + (1e6 / timeUsPerQuery) + " query/s");
        System.out.println("Total Search Time = " + (totalUsertime / 1e6f) + " sec");
        System.out.println("Error Bound Avg = " + (errorBoundAvg / queryVectors.size()));
        System.out.println("Max Estimator Size = " + maxEstimatorSize);
        System.out.println("Total Estimator Adds = " + totalEstimatorAdds);
    }
}
