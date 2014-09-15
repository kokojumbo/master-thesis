package com.master.thesis.utils;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

/**
 * Created by Marcin Gumkowski on 26.06.14.
 */
public class LoadUtils {

    public static Instances loadDataFile(String filename) {
        Instances data;
        String path = System.getProperty("user.dir") + "\\resources\\datasets\\";
        path = path.concat(filename);
//        System.out.println("Path:\t\t" + path);
//        System.out.println("Dataset:\t" + filename);
        ConverterUtils.DataSource source;
        try {
            source = new ConverterUtils.DataSource(path);
            data = source.getDataSet();
            //System.out.println(filename + " -> Data loaded.");
            // Normalizacja atrybutów, domyslne ustawienia
            Normalize filterNorm = new Normalize();
            filterNorm.setInputFormat(data);
            data = Filter.useFilter(data, filterNorm);
            //System.out.println("Data Normalized");
            //System.out.println();
            return data;
        } catch (Exception e) {
            e.printStackTrace();

        }
        return null;

    }

    public static Instances loadDataFilePrint(String filename) {
        Instances data;
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
}
