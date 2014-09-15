package com.master.thesis.deprecated;


/**
 * Created by Marcin Gumkowski on 03.03.14.
 */
public class MainXmeans {

    public static void main(String[] args) {
        String[] fileNames = {"acl.arff", "new-thyroid.arff", "vehicle.arff", "ecoli.arff", "ionosphere.arff", "haberman.arff", "abalone.arff", "transfusion.arff", "yeast-ME2.arff"};

        runner(fileNames);

    }


    public static void runner(String[] fileNames){

        for (String filename : fileNames) {
            System.out.println("======================================================================");
            System.out.println("======================================================================");
            System.out.println("======================================================================");
            System.out.println("======================      "+filename+"       =======================");
            System.out.println("======================================================================");
            System.out.println("======================================================================");
            System.out.println("======================================================================");
            ClusterImbalancedAlgorithmXMeans ciax = new ClusterImbalancedAlgorithmXMeans();
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


}
