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
        int kTotalClusters = Integer.parseInt(args[2]);
        int dimensions = Integer.parseInt(args[3]);
        int B_QUERY = Integer.parseInt(args[4]);
        String resultPath = args[5];  // eg "/Users/jwagster/Desktop/gist1m/ivfrn_output/"

        // FIXME: FUTURE - clean up these constants
        int C = kTotalClusters;
        int D = dimensions;
        int B = (D + 63) / 64 * 64;

        // FIXME: FUTURE - clean up gross path mgmt
        String queryPath = String.format("%s%s_query.fvecs", source, dataset);
        float[][] Q = IOUtils.readFvecs(new FileInputStream(queryPath));

        String dataPath = String.format("%s%s_base.fvecs", source, dataset);
        float[][] X = IOUtils.readFvecs(new FileInputStream(dataPath));

        String groundTruthPath = String.format("%s%s_groundtruth.ivecs", source, dataset);
        int[][] G = IOUtils.readIvecs(new FileInputStream(groundTruthPath));

        String transformationPath = String.format("%sP_C%d_B%d.fvecs", source, C, B);
        float[][] P = IOUtils.readFvecs(new FileInputStream(transformationPath));

        String indexPath = String.format("%sivfrabitq%d_B%d.index", source, C, B);

        // Write to file
        FileWriter writer = new FileWriter(new File(resultPath + "/result.txt"));
        writer.write(Q.toString() + "\n");
        writer.close();

        IVFRN ivfrn = IVFRN.load(indexPath);
        float[][] RandQ = MatrixUtils.multiply(Q, P);

        test(kTotalClusters, Q, RandQ, X, G, ivfrn, B_QUERY);

    }

    public static void test(int k, float[][] Q, float[][] RandQ, float[][] X, int[][] G, IVFRN ivf, int B_QUERY) {
        int nprobes = 300; // FIXME: FUTURE - hardcoded

        float totalUsertime = 0;
        float totalRatio = 0;
        int correctCount = 0;

        for (int i = 0; i < Q.length; i++) {
            long startTime = System.nanoTime();
            float distK = Float.POSITIVE_INFINITY;
            PriorityQueue<Result> KNNs = ivf.search(Q[i], RandQ[i], k, nprobes, distK, B_QUERY);
            long endTime = System.nanoTime();
            float usertime = (endTime - startTime) / 1e6f;
            totalUsertime += usertime;

            float ratio = Utils.getRatio(i, Q, X, G, KNNs);
            totalRatio += ratio;

            int correct = 0;
            while (!KNNs.isEmpty()) {
                int id = KNNs.remove().c();
                for (int j = 0; j < k; j++) {
                    if (id == G[i][j]) correct++;
                }
            }
            correctCount += correct;
            // FIXME: FUTURE - use logging instead
            System.out.println("recall = " + correct + " / " + k + " " + i + 1 + " / " + Q.length + " " + usertime + "us");
        }

        // FIXME: FUTURE - missing rotation time?
        float timeUsPerQuery = totalUsertime / Q.length;
        float recall = (float) correctCount / (Q.length * k);
        float averageRatio = totalRatio / (Q.length * k);

        // FIXME: FUTURE - logs instead of println
        System.out.println("------------------------------------------------");
        System.out.println("nprobe = " + nprobes + " k = " + k);
        System.out.println("Recall = " + recall * 100000f + "%\t" + "Ratio = " + averageRatio);
        System.out.println("Time = " + timeUsPerQuery + " us\t QPS = " + (1e6 / timeUsPerQuery) + " query/s");
    }
}
