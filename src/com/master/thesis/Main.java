package com.master.thesis;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;


/**
 * Created by Marcin Gumkowski on 03.03.14.
 */
public class Main {

    public static void main(String[] args) {


        System.out.println("test");
        System.out.println(System.getProperty("user.dir"));

        String path = System.getProperty("user.dir") + "\\resources\\datasets\\";
        path = path.concat("abalone.arff");
        System.out.println(path);
        DataSource source = null;
        try {

            source = new DataSource(path);
            Instances data = source.getDataSet();
            // setting class attribute if the data format does not provide this information
            // For example, the XRFF format saves the class attribute information as well

            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

        System.out.println(data.numClasses());
        } catch (Exception e) {

            e.printStackTrace();
        }

        //data1
        //data2
        //dbscan on data1


        //wczytac arff
        //Rozdzielic na dwie klasy
        //DB scan dla mniejszosciowej
        //Naniesc przyklady z klasy wiekszosciowej tam gdzie wpdaja do skupisk
        //Rozdzielic skupiska
        //Dla kazdego skupiska użyc klasyfikatora

    }
}
