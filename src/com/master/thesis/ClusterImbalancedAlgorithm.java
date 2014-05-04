package com.master.thesis;

import weka.clusterers.AbstractClusterer;
import weka.clusterers.ClusterEvaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.Map;
import java.util.TreeMap;

/**
 * Created by Marcin Gumkowski on 04.05.14.
 */
public class ClusterImbalancedAlgorithm implements ImbalancedAlgorithm {

    private Instances data;

    private AbstractClusterer clusterer;

    @Override
    public void start() throws Exception {
        if (clusterer == null || data == null) {
            throw new Exception("Please provide data examples and clusterer.");
        }

        // Przygotuj dane -> Ustaw klasę decyzyjną
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // Rozdziel przykłady na dwie klasy: mniejszościową i większościową
        Instances minorityInstances = new Instances(data);
        Instances majorityInstances = new Instances(data);
        separateDecisionClasses(data, minorityInstances, majorityInstances);

        // Pokaż statystyki przykladow
        showInitialDataStatitistics(data, minorityInstances, majorityInstances);

        // Usuń klasę decyzyjną -> uczenie nienadzorowane
        weka.filters.unsupervised.attribute.Remove filter = new weka.filters.unsupervised.attribute.Remove();
        filter.setAttributeIndices("" + (minorityInstances.classIndex() + 1));
        filter.setInputFormat(minorityInstances);
        Instances minorityInstancesNoClass = Filter.useFilter(minorityInstances, filter);
        Instances majorityInstancesNoClass = Filter.useFilter(majorityInstances, filter);
        Instances dataNoClass = Filter.useFilter(data, filter);

        // Stwórz skupiska
        clusterer.buildClusterer(minorityInstancesNoClass);

        // Pokaż wyniki analizy skupisk
        ClusterEvaluation evaluation = new ClusterEvaluation();
        evaluation.setClusterer(clusterer);
        evaluation.evaluateClusterer(minorityInstancesNoClass);
        System.out.println(evaluation.clusterResultsToString());

        // Klasyfikuj przyklady klasy wiekszosciowej do skupisk (nie modyfukuje skupisk)
        Map<Integer, Integer> majorityHistogram = createHistogram(clusterer, majorityInstancesNoClass);

        Map<Integer, Integer> minorityHistogram = createHistogram(clusterer, minorityInstancesNoClass);

        // Pokaz statystyki
        showHistogram(majorityHistogram, "Majority");
        showHistogram(minorityHistogram, "Minority");
        showCombinedHistograms(majorityHistogram, minorityHistogram);

        // Rozdziel przyklady z klasami decyzyjnymi wedlug wyznaczonych skupisk
        Map<Integer, Instances> clustersAssignmentsMap = createClusterAssignmentsMap(data, dataNoClass, clusterer);
        showClusterAssignmentsMap(clustersAssignmentsMap);

        // TODO można zrobić SMOTE lub juz próbować klasyfikować

    }

    @Override
    public Instances loadDataFile(String filename) {

        String path = System.getProperty("user.dir") + "\\resources\\datasets\\";
        path = path.concat(filename);
        System.out.println("Path:\t\t" + path);
        System.out.println("Dataset:\t" + filename);
        ConverterUtils.DataSource source;
        try {
            source = new ConverterUtils.DataSource(path);
            data = source.getDataSet();
            System.out.println(filename + " -> Data loaded.");
            // Normalizacja atrybutów, domyslne ustawienia
            Normalize filterNorm = new Normalize();
            filterNorm.setInputFormat(data);
            data = Filter.useFilter(data, filterNorm);
            System.out.println("Data Normalized");
            System.out.println();
            return data;
        } catch (Exception e) {
            e.printStackTrace();

        }
        return null;

    }

    @Override
    public void setClusterer(AbstractClusterer clusteringAlgorithm) {
        this.clusterer = clusteringAlgorithm;
    }

    @Override
    public void setClustererOptions(String[] options) {

    }

    @Override
    public void setFilters() {

    }

    @Override
    public double getMinorityValue(Instances instances) throws Exception {
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

    private void separateDecisionClasses(Instances instances, Instances minorityInstances, Instances majorityInstances) throws Exception {
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

    private void showHistogram(Map<Integer, Integer> histogram, String name) {
        System.out.println();
        System.out.println(name + " histogram");
        for (Map.Entry<Integer, Integer> entry : histogram.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
        System.out.println();
    }

    private void showCombinedHistograms(Map<Integer, Integer> majorityHistogram, Map<Integer, Integer> minorityHistogram) {
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

    private Map<Integer, Integer> createHistogram(AbstractClusterer clusterer, Instances instances) {
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

    private Map<Integer, Integer> createHistogram(double[] assignments) {
        Map<Integer, Integer> histogram = new TreeMap<Integer, Integer>();
        for (double val : assignments) {
            histogram.put((int) val, histogram.containsKey((int) val) ? histogram.get((int) val) + 1 : 1);
        }
        return histogram;
    }

    private Map<Integer, Instances> createClusterAssignmentsMap(Instances data, Instances dataNoClass, AbstractClusterer clusterer) {
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

    private void showClusterAssignmentsMap(Map<Integer, Instances> clustersAssignmentsMap) throws Exception {
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


    private void showInitialDataStatitistics(Instances baseData, Instances minorityInstances, Instances majorityInstances) {
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
}