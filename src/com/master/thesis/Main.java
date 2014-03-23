package com.master.thesis;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.DBSCAN;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;

import java.util.Map;
import java.util.TreeMap;


/**
 * Created by Marcin Gumkowski on 03.03.14.
 */
public class Main {

    public static void main(String[] args) {


        ClusterImbalancedAlgorithm cia = new ClusterImbalancedAlgorithm();
        cia.loadDataFile("transfusion.arff");
        try {
            cia.start();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }




}
