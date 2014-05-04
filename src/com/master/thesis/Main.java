package com.master.thesis;


import weka.clusterers.SimpleKMeans;

/**
 * Created by Marcin Gumkowski on 03.03.14.
 */
public class Main {

    public static void main(String[] args) {
        ClusterImbalancedAlgorithm cia = new ClusterImbalancedAlgorithm();
        cia.loadDataFile("cmc.arff");
        SimpleKMeans skm = new SimpleKMeans();
        try {
            skm.setOptions(weka.core.Utils.splitOptions("-N 3 -A \"weka.core.EuclideanDistance -R first-last\" -I 500 -S 10"));
//            DBSCAN dbScan = new DBSCAN();
//            dbScan.setDatabase_Type("weka.clusterers.forOPTICSAndDBScan.Databases.SequentialDatabase");
//            dbScan.setDatabase_distanceType("weka.clusterers.forOPTICSAndDBScan.DataObjects.EuclideanDataObject");
//            dbScan.setEpsilon(0.5);
//            dbScan.setMinPoints(4);
            cia.setClusterer(skm);
            cia.start();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }


}
