package com.master.thesis;


import weka.clusterers.SimpleKMeans;

/**
 * Created by Marcin Gumkowski on 03.03.14.
 */
public class Main {

    public static void main(String[] args) {
//        ClusterImbalancedAlgorithm cia = new ClusterImbalancedAlgorithm();
//        String filename = "acl.arff";
//        cia.loadDataFile(filename);
//
//        try {
//            SimpleKMeans skm = new SimpleKMeans();
//            skm.setOptions(weka.core.Utils.splitOptions("-N 3 -A \"weka.core.EuclideanDistance -R first-last\" -I 500 -S 10"));
//            cia.setClusterer(skm);
//            cia.setAutoParameterization(true);
//            cia.smoteEnable(true);
//            cia.setClassifier("weka.classifiers.trees.J48", weka.core.Utils.splitOptions("-U -M 2"));
//            cia.start();
//        } catch (Exception e) {
//            e.printStackTrace();
//        }

        ClusterImbalancedAlgorithmXMeans ciax = new ClusterImbalancedAlgorithmXMeans();
        String filename = "abalone.arff";
        ciax.loadDataFile(filename);

        try {

            ciax.setAutoParameterization(true);
            ciax.smoteEnable(true);
            ciax.setClassifier("weka.classifiers.trees.J48", weka.core.Utils.splitOptions("-U -M 2"));
            ciax.start();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


}
