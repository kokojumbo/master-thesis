package com.master.thesis;


import weka.clusterers.SimpleKMeans;

/**
 * Created by Marcin Gumkowski on 03.03.14.
 */
public class Main {

    public static void main(String[] args) {
        ClusterImbalancedAlgorithm cia = new ClusterImbalancedAlgorithm();
        cia.loadDataFile("abalone.arff");
        SimpleKMeans skm = new SimpleKMeans();
        try {
            skm.setOptions(weka.core.Utils.splitOptions("-N 3 -A \"weka.core.EuclideanDistance -R first-last\" -I 500 -S 10"));
            cia.setClusterer(skm);
            cia.smoteEnable(true);
            cia.setClassiffier("weka.classifiers.trees.J48", null);
            cia.start();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }


}
