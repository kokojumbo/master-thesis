package com.master.thesis;

import com.sun.org.apache.xpath.internal.SourceTree;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.DBSCAN;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;

import java.util.Map;
import java.util.TreeMap;

/**
 * Created by Marcin Gumkowski on 23.03.14.
 */
public class ClusterImbalancedAlgorithm implements ImbalancedAlgorithm {


    private Instances data;

    @Override
    public void setParameters() {

    }

    @Override
    public void getParameters() {

    }

    @Override
    public void start() throws Exception {
        // Przygotuj dane -> Ustaw klasę decyzyjną
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        System.out.println("=====================================");


        System.out.println(String.format("%-20s %s", "    ", "Count"));
        System.out.println("-------------------------------------");
        System.out.println(String.format("%-20s %s", "Instances", data.numInstances()));
        System.out.println(String.format("%-20s %s", "Atrributes", data.numAttributes()));
        System.out.println(String.format("%-20s %s", "Distinct Values", data.numDistinctValues(data.classAttribute())));


        // Rozdziel przykłady na dwie klasy: mniejszościową i większościową
        Instances minorityInstances = new Instances(data);
        Instances majorityInstances = new Instances(data);
        separateDecisionClasses(data, minorityInstances, majorityInstances);

        System.out.println(String.format("%-20s %s", "Minority Class", minorityInstances.numInstances()));
        System.out.println(String.format("%-20s %s", "Majority Class", majorityInstances.numInstances()));
        System.out.println("=====================================");
        System.out.println();

        // Wyszukaj skupiska w klasie mniejszościowej

        // Ustaw opcje DBScan'a
        DBSCAN dbScan = new DBSCAN();
        dbScan.setDatabase_Type("weka.clusterers.forOPTICSAndDBScan.Databases.SequentialDatabase");
        dbScan.setDatabase_distanceType("weka.clusterers.forOPTICSAndDBScan.DataObjects.EuclideanDataObject");
        dbScan.setEpsilon(0.5);
        dbScan.setMinPoints(4);

        // Usuń klasę decyzyjną -> uczenie nienadzorowane
        weka.filters.unsupervised.attribute.Remove filter = new weka.filters.unsupervised.attribute.Remove();
        filter.setAttributeIndices("" + (minorityInstances.classIndex() + 1));
        filter.setInputFormat(minorityInstances);
        Instances minorityInstancesNoClass = Filter.useFilter(minorityInstances, filter);


        // Stwórz skupiska
        dbScan.buildClusterer(minorityInstancesNoClass);

        // Pokaż wyniki
        ClusterEvaluation evaluation = new ClusterEvaluation();
        evaluation.setClusterer(dbScan);

        evaluation.evaluateClusterer(minorityInstancesNoClass);
        System.out.println(evaluation.clusterResultsToString());


        Map<Integer, Integer> histogram = new TreeMap<Integer, Integer>();
        for (int i = 0; i < majorityInstances.numInstances(); i++) {
            Instance currInst = majorityInstances.instance(i);
            try {
                int cluster = dbScan.clusterInstance(currInst);

                histogram.put(cluster, histogram.containsKey(cluster) ? histogram.get(cluster) + 1 : 0);
                //System.out.println(dbScan.clusterInstance(currInst));

            } catch (Exception e) {

                //System.out.println("NOISE -> Cannot be clustered");
                histogram.put(-1, histogram.containsKey(-1) ? histogram.get(-1) + 1 : 0);

            }

        }
        System.out.println();

        for (Map.Entry<Integer, Integer> entry : histogram.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }


    }

    @Override
    public Instances loadDataFile(String filename) {

        String path = System.getProperty("user.dir") + "\\resources\\datasets\\";
        path = path.concat(filename);
        System.out.println("Path:\t\t" + path);
        System.out.println("Dataset:\t" + filename);
        DataSource source;
        try {
            source = new DataSource(path);
            data = source.getDataSet();
            System.out.println(filename + " -> Data loaded.");
            return data;
        } catch (Exception e) {
            e.printStackTrace();

        }
        return null;

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


    public void separateDecisionClasses(Instances instances, Instances minorityInstances, Instances majorityInstances) throws Exception {
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
}
