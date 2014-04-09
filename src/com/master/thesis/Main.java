package com.master.thesis;


/**
 * Created by Marcin Gumkowski on 03.03.14.
 */
public class Main {

    public static void main(String[] args) {
       DBScanImbalancedAlgorithm dia = new DBScanImbalancedAlgorithm();
        dia.loadDataFile("cmc.arff");
        try {
            dia.start();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }


}
