package com.master.thesis;

import weka.clusterers.AbstractClusterer;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Map;
import java.util.TreeMap;

/**
 * Created by Marcin Gumkowski on 14.05.14.
 */
public class ImbalancedUtils {
    public static double getMinorityValue(Instances instances) throws Exception {
        int numInstances = instances.numInstances();
        if (numInstances < 1) {
            throw new Exception("Empty Instances");
        }
        Map<String, Integer> mapValues = new TreeMap<String, Integer>();
        for (int i = 0; i < numInstances; i++) {
            Instance instance = instances.instance(i);
            String key = String.valueOf(instance.value(instance.classAttribute()));
            if (mapValues.containsKey(key)) {
                mapValues.put(String.valueOf(key), (mapValues.get(key) + 1));
            } else {
                mapValues.put(String.valueOf(key), 1);
            }
        }
        mapValues = MapUtil.sortByValue(mapValues);
        for (Map.Entry<String, Integer> entry : mapValues.entrySet()) {
            //System.out.println(entry.getKey() + " " + entry.getValue());
            return Double.valueOf(entry.getKey());
        }

        return -1.0;
    }

    public static void separateDecisionClasses(Instances instances, Instances minorityInstances, Instances majorityInstances) throws Exception {
        Double minorityValue = getMinorityValue(instances);
        minorityInstances.delete();
        majorityInstances.delete();
        int numInstances = instances.numInstances();
        for (int i = 0; i < numInstances; i++) {
            Instance instance = instances.instance(i);
            if (instance.value(instance.classAttribute()) == minorityValue) {
                minorityInstances.add(instance);
            } else {
                majorityInstances.add(instance);
            }
        }


    }

    public static void showHistogram(Map<Integer, Integer> histogram, String name) {
        System.out.println();
        System.out.println(name + " histogram");
        for (Map.Entry<Integer, Integer> entry : histogram.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
        System.out.println();
    }

    public static void showCombinedHistograms(Map<Integer, Integer> majorityHistogram, Map<Integer, Integer> minorityHistogram) {
        System.out.println();
        System.out.println("Balance in clusters");
        double averageBalance = 0.0;
        int total = 0;
        for (Map.Entry<Integer, Integer> entry : majorityHistogram.entrySet()) {
            int cluster = entry.getKey();
            int majorityCount = entry.getValue();
            int minorityCount = minorityHistogram.get(entry.getKey());
            int sum = majorityCount + minorityCount;
            total += sum;
            double balance = 100.0 * minorityCount / (minorityCount + majorityCount);
            averageBalance += balance;
            System.out.print(cluster + ": " + majorityCount + ", " + minorityCount + " ->\t" + sum + " ");
            System.out.println(String.format("\t\t%.2f %%", balance));

        }
        averageBalance /= majorityHistogram.size();
        System.out.println(String.format("Average balance:\t %.2f %%", averageBalance));
        System.out.println(String.format("Total examples:\t\t %d", total));
        System.out.println();
    }

    public static Map<Integer, Integer> createHistogram(AbstractClusterer clusterer, Instances instances) {
        Map<Integer, Integer> histogram = new TreeMap<Integer, Integer>();
        for (int i = 0; i < instances.numInstances(); i++) {
            Instance currInst = instances.instance(i);
            try {
                int cluster = clusterer.clusterInstance(currInst);

                histogram.put(cluster, histogram.containsKey(cluster) ? histogram.get(cluster) + 1 : 1);
            } catch (Exception e) {
                // Noise
                //e.printStackTrace();
                histogram.put(-1, histogram.containsKey(-1) ? histogram.get(-1) + 1 : 1);

            }

        }
        return histogram;
    }

    public static Map<Integer, Integer> createHistogram(double[] assignments) {
        Map<Integer, Integer> histogram = new TreeMap<Integer, Integer>();
        for (double val : assignments) {
            histogram.put((int) val, histogram.containsKey((int) val) ? histogram.get((int) val) + 1 : 1);
        }
        return histogram;
    }

    public static Map<Integer, Instances> createClusterAssignmentsMap(Instances data, Instances dataNoClass, AbstractClusterer clusterer) {
        Map<Integer, Instances> clustersAssignmentsMap = new TreeMap<Integer, Instances>();
        for (int i = 0; i < data.numInstances(); i++) {
            Instance currInst = dataNoClass.instance(i);
            try {
                int cluster = clusterer.clusterInstance(currInst);

                if (clustersAssignmentsMap.containsKey(cluster)) {
                    clustersAssignmentsMap.get(cluster).add(data.instance(i));

                } else {
                    Instances dataClustered = new Instances(data);
                    dataClustered.delete();
                    dataClustered.add(data.instance(i));
                    clustersAssignmentsMap.put(cluster, dataClustered);
                }

            } catch (Exception e) {
                // Noise
                //e.printStackTrace();
                if (clustersAssignmentsMap.containsKey(-1)) {
                    clustersAssignmentsMap.get(-1).add(data.instance(i));

                } else {
                    Instances dataClustered = new Instances(data);
                    dataClustered.delete();
                    dataClustered.add(data.instance(i));
                    clustersAssignmentsMap.put(-1, dataClustered);
                }

            }

        }
        return clustersAssignmentsMap;
    }

    public static void showClusterAssignmentsMap(Map<Integer, Instances> clustersAssignmentsMap) throws Exception {
        System.out.println();
        for (Map.Entry<Integer, Instances> entry : clustersAssignmentsMap.entrySet()) {
            System.out.println("Cluster -> " + entry.getKey());

            // Rozdziel przykłady na dwie klasy: mniejszościową i większościową
            Instances minorityInstancesInCluster = new Instances(entry.getValue());
            Instances majorityInstancesInCluster = new Instances(entry.getValue());
            separateDecisionClasses(entry.getValue(), minorityInstancesInCluster, majorityInstancesInCluster);

            // Pokaż statystyki przykladow
            showInitialDataStatitistics(entry.getValue(), minorityInstancesInCluster, majorityInstancesInCluster);

        }
        System.out.println();
    }

    public static void showInitialDataStatitistics(Instances baseData, Instances minorityInstances, Instances majorityInstances) {
        System.out.println("=====================================");
        System.out.println(String.format("%-20s %s", "    ", "Count"));
        System.out.println("-------------------------------------");
        System.out.println(String.format("%-20s %s", "Instances", baseData.numInstances()));
        System.out.println(String.format("%-20s %s", "Atrributes", baseData.numAttributes()));
        System.out.println(String.format("%-20s %s", "Distinct Values", baseData.numDistinctValues(baseData.classAttribute())));
        System.out.println(String.format("%-20s %s", "Minority Class", minorityInstances.numInstances()));
        System.out.println(String.format("%-20s %s", "Majority Class", majorityInstances.numInstances()));
        System.out.println(String.format("%-20s %.2f %%", "Balance", 100.0 * minorityInstances.numInstances() / baseData.numInstances()));
        System.out.println("=====================================");
        System.out.println();
    }

    public static String getParametersForKMeans(Instances instances, String name) {

        SimpleKMeans kmeans = new SimpleKMeans();
        String options = "-init 0 -max-candidates 100 -periodic-pruning 10000 -min-density 2.0 -t1 -1.25 -t2 -1.0 -V -N 2 -A \"weka.core.EuclideanDistance -R first-last\" -I 500 -num-slots 1 -S 10";
        try {
            kmeans.setOptions(weka.core.Utils.splitOptions(options));
            int numK = 20;
            double[] results = new double[numK];

            for (int i = 1; i <= numK; i++) {
                kmeans.setNumClusters(i);
                kmeans.buildClusterer(instances);


                double[][] centroids = new double[i][kmeans.getClusterCentroids().get(0).numAttributes()];


                double[][] stdDevs = new double[i][kmeans.getClusterStandardDevs().get(0).numAttributes()];

                for (int j = 0; j < i; j++) {
                    for (int k = 0; k < kmeans.getClusterCentroids().get(0).numAttributes(); k++) {
                        if (!Double.isNaN(kmeans.getClusterCentroids().get(j).value(k)) && !Double.isNaN(kmeans.getClusterStandardDevs().get(j).value(k))) {
                            centroids[j][k] = kmeans.getClusterCentroids().get(j).value(k);
                            stdDevs[j][k] = kmeans.getClusterStandardDevs().get(j).value(k);
                        } else {
                            centroids[j][k] = 0.0;
                            stdDevs[j][k] = 0.0;
                        }

                    }
                }


                double sum_clust = 0;
                for (int j = 0; j < centroids.length; j++) {
                    double sum_att = 0;
                    for (int k = 0; k < centroids[0].length; k++) {
                        double d = stdDevs[j][k];
                        sum_att += d;
                    }
                    sum_clust += sum_att;
                }
                results[i - 1] = sum_clust / centroids.length;

                // Pokaż wyniki analizy skupisk
                ClusterEvaluation evaluation = new ClusterEvaluation();
                evaluation.setClusterer(kmeans);
                evaluation.evaluateClusterer(instances);

                //System.out.println(evaluation.clusterResultsToString());
            }
            System.out.println(name);
            for (int i = 0; i < numK; i++) {
                System.out.println(String.format("%.3f", results[i]));
            }
            System.out.println();
        } catch (Exception e) {
            e.printStackTrace();
        }


        return "";

    }

}
