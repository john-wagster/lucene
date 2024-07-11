package org.apache.lucene.sandbox.rabitq;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.util.PriorityQueue;

public class Search {

    public static void main(String[] args) throws Exception {
        // FIXME: FUTURE - use fastscan to search over the ivfrn?
        // FIXME: FUTURE - get metrics setup appropriately as needed
        // FIXME: FUTURE - better arg parsing

        // search DIRECTORY_TO_DATASET DATASET_NAME NUM_CENTROIDS DIMENSIONS B_QUERY OUTPUT_PATH
        String source = args[0];  // eg "/Users/jwagster/Desktop/gist1m/gist/"
        String dataset = args[1]; // eg "gist"
        int numCentroids = Integer.parseInt(args[2]);
        int dimensions = Integer.parseInt(args[3]);
        int B_QUERY = Integer.parseInt(args[4]);
        int k = Integer.parseInt(args[5]);
        String resultPath = args[6];  // eg "/Users/jwagster/Desktop/gist1m/ivfrn_output/"

        // FIXME: FUTURE - clean up these constants
        int B = (dimensions + 63) / 64 * 64;

        // FIXME: FUTURE - clean up gross path mgmt
        String queryPath = String.format("%s%s_query.fvecs", source, dataset);
        float[][] Q = IOUtils.readFvecs(new FileInputStream(queryPath));

        String dataPath = String.format("%s%s_base.fvecs", source, dataset);
        float[][] X = IOUtils.readFvecs(new FileInputStream(dataPath));

        String groundTruthPath = String.format("%s%s_groundtruth.ivecs", source, dataset);
        int[][] G = IOUtils.readIvecs(new FileInputStream(groundTruthPath));

        String transformationPath = String.format("%sP_C%d_B%d.fvecs", source, numCentroids, B);
        float[][] P = IOUtils.readFvecs(new FileInputStream(transformationPath));

        String indexPath = String.format("%sivfrabitq%d_B%d.index", source, numCentroids, B);

        // Write to file
        FileWriter writer = new FileWriter(new File(resultPath + "/result.txt"));
        writer.write(Q.toString() + "\n");
        writer.close();

        IVFRN ivfrn = IVFRN.load(indexPath);
        float[][] RandQ = MatrixUtils.multiply(Q, P);

        test(Q, RandQ, X, G, ivfrn, k, B_QUERY);
    }

    public static void test(float[][] Q, float[][] RandQ, float[][] X, int[][] G, IVFRN ivf, int k, int B_QUERY) {

        int nprobes = 300;
        nprobes = Math.min(nprobes, ivf.C); // FIXME: FUTURE - hardcoded
        assert nprobes <= k;

        float totalUsertime = 0;
        float totalRatio = 0;
        int correctCount = 0;

        float errorBoundAvg = 0f;
        int totalExploredNNs = 0;
        int totalComparisons = 0;
        int maxEstimatorSize = 0;
        int totalEstimatorAdds = 0;
        System.out.println("Starting search");
        for (int i = 0; i < Q.length; i++) {
            long startTime = System.nanoTime();
            IVFRNResult result = ivf.search(Q[i], RandQ[i], k, nprobes, B_QUERY);
            PriorityQueue<Result> KNNs = result.results();
            IVFRNStats stats = result.stats();
            float usertime = (System.nanoTime() - startTime) / 1e3f; // convert to microseconds to compare to the c impl
            totalUsertime += usertime;

            PriorityQueue<Result> copyOfKNN = new PriorityQueue<>();
            copyOfKNN.addAll(KNNs);
            float ratio = Utils.getRatio(i, Q, X, G, copyOfKNN);
            totalRatio += ratio;

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
            // FIXME: FUTURE - use logging instead
//            System.out.println("recall = " + correct + " / " + k + " " + (i + 1) + " / " + Q.length + " " + usertime + " us" + " err bound avg = " + stats.errorBoundAvg() + " nn explored = " + stats.totalExploredNNs());

            if (i % 1500 == 0) {
                System.out.print(".");
            }

            errorBoundAvg += stats.errorBoundAvg();
            totalExploredNNs += stats.totalExploredNNs();
            totalComparisons += stats.totalComparisons();
            maxEstimatorSize = stats.maxEstimatorSize();
            totalEstimatorAdds += stats.totalEstimatorQueueAdds();
        }
        System.out.println();

        // FIXME: FUTURE - missing rotation time?
        float timeUsPerQuery = totalUsertime / Q.length;
        float recall = (float) correctCount / (Q.length * k);
        float averageRatio = totalRatio / (Q.length * k);

        // FIXME: FUTURE - logs instead of println
        System.out.println("------------------------------------------------");
        System.out.println("nprobe = " + nprobes + "\tk = " + k + "\tCoarse Clusters = " + ivf.centroids.length);
        System.out.println("Recall = " + recall * 100f + "%\t" + "Ratio = " + averageRatio);
        System.out.println("Avg Time Per Search = " + timeUsPerQuery + " us\t QPS = " + (1e6 / timeUsPerQuery) + " query/s");
        System.out.println("Total Search Time = " + (totalUsertime / 1e6f) + " sec");
        System.out.println("Error Bound Avg = " + (errorBoundAvg / Q.length));
        System.out.println("Total Explored KNN = " + totalExploredNNs);
        System.out.println("Total Comparisons = " + totalComparisons);
        System.out.println("Max Estimator Size = " + maxEstimatorSize);
        System.out.println("Total Estimator Adds = " + totalEstimatorAdds);
    }
}
