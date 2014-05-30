package com.master.thesis;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.clusterers.AbstractClusterer;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

import static com.master.thesis.ImbalancedUtils.getParametersForKMeans;
import static com.master.thesis.ImbalancedUtils.separateDecisionClasses;

/**
 * Created by Marcin Gumkowski on 04.05.14.
 */
public class ClusterImbalancedAlgorithmParamTest implements ImbalancedAlgorithm {

    private Instances data;

    private AbstractClusterer clusterer;

    private String filename;

    private boolean smoteEnable;

    private boolean autoParametrizationEnable;


    private Classifier classifier;

    @Override
    public void start() throws Exception {
        if (clusterer == null || data == null) {
            throw new Exception("Please provide data examples and clusterer.");
        }

        // Przygotuj dane -> Ustaw klasę decyzyjną
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // Rozdziel przykłady na dwie klasy: mniejszościową i większościową
        Instances minorityInstances = new Instances(data);
        Instances majorityInstances = new Instances(data);
        separateDecisionClasses(data, minorityInstances, majorityInstances);

        // Pokaż statystyki przykladow
        //showInitialDataStatitistics(data, minorityInstances, majorityInstances);

        // Usuń klasę decyzyjną -> uczenie nienadzorowane
        weka.filters.unsupervised.attribute.Remove filter = new weka.filters.unsupervised.attribute.Remove();
        filter.setAttributeIndices("" + (minorityInstances.classIndex() + 1));
        filter.setInputFormat(minorityInstances);
        Instances minorityInstancesNoClass = Filter.useFilter(minorityInstances, filter);
        Instances majorityInstancesNoClass = Filter.useFilter(majorityInstances, filter);
        Instances dataNoClass = Filter.useFilter(data, filter);

        // Stwórz skupiska
        String options;
        if (autoParametrizationEnable) {
            options = getParametersForKMeans(minorityInstancesNoClass, filename);
            clusterer.setOptions(weka.core.Utils.splitOptions(options));
        }


    }

    @Override
    public Instances loadDataFile(String filename) {
        this.filename = filename;
        String path = System.getProperty("user.dir") + "\\resources\\datasets\\";
        path = path.concat(filename);
        //System.out.println("Path:\t\t" + path);
        //.out.println("Dataset:\t" + filename);
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

    @Override
    public void setClusterer(AbstractClusterer clusteringAlgorithm) {


        this.clusterer = clusteringAlgorithm;

    }

    @Override
    public void smoteEnable(boolean enable) {
        this.smoteEnable = enable;
    }

    @Override
    public void setClassifier(String classifier, String[] classifierOptions) {
        try {
            this.classifier = AbstractClassifier.forName(classifier, classifierOptions);
        } catch (Exception e) {
            e.printStackTrace();
        }
        ;
    }

    @Override
    public void setAutoParameterization(boolean enable) {
        this.autoParametrizationEnable = enable;
    }


    private double getSMOTEPercentage(Instances instances) {
        Instances minorityInstances = new Instances(instances);
        Instances majorityInstances = new Instances(instances);
        Instances newInstances = new Instances(instances);
        try {
            separateDecisionClasses(newInstances, minorityInstances, majorityInstances);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return ((1.0 * majorityInstances.numInstances() / minorityInstances.numInstances()) - 1) * 100.0;


    }

}
