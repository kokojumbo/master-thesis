package com.master.thesis;

import weka.clusterers.SimpleKMeans;

import java.io.File;

/**
 * Created by Marcin Gumkowski on 29.05.14.
 */
public class KMeansParameters {


    public static void main(String[] args) {

        try {

            String path = System.getProperty("user.dir") + "\\resources\\datasets\\";

            File folder = new File(path);
            File[] listOfFiles = folder.listFiles();

            for (int i = 0; i < listOfFiles.length; i++) {
                if (listOfFiles[i].isFile()) {
                    System.out.println("File " + listOfFiles[i].getName());
                } else if (listOfFiles[i].isDirectory()) {
                    System.out.println("Directory " + listOfFiles[i].getName());
                }
            }

            for (int i = 0; i < listOfFiles.length; i++) {
                ClusterImbalancedAlgorithmParamTest test = new ClusterImbalancedAlgorithmParamTest();
                test.loadDataFile(listOfFiles[i].getName());
                SimpleKMeans skm = new SimpleKMeans();
                skm.setOptions(weka.core.Utils.splitOptions("-N 3 -A \"weka.core.EuclideanDistance -R first-last\" -I 500 -S 10"));
                test.setClusterer(skm);
                test.setAutoParameterization(true);
                test.smoteEnable(true);
                test.setClassifier("weka.classifiers.trees.J48", weka.core.Utils.splitOptions("-U -M 2"));

                test.start();


            }





        } catch (Exception e) {
            //e.printStackTrace();
        }


    }


}
