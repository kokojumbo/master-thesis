package com.master.thesis.deprecated;

import com.master.thesis.core.ImbalancedAlgorithm;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.clusterers.AbstractClusterer;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.XMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.Map;
import java.util.Random;

import static com.master.thesis.utils.ImbalancedUtils.*;

/**
 * Created by Marcin Gumkowski on 04.05.14.
 */
public class ClusterImbalancedAlgorithmXMeans implements ImbalancedAlgorithm {

    private Instances data;

    private AbstractClusterer clusterer;

    private String filename;

    private boolean smoteEnable;

    private boolean autoParametrizationEnable;


    private Classifier classifier;

    @Override
    public void start() throws Exception {
//        if (clusterer == null || data == null) {
//            throw new Exception("Please provide data examples and clusterer.");
//        }

        // Przygotuj dane -> Ustaw klasę decyzyjną
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // Rozdziel przykłady na dwie klasy: mniejszościową i większościową
        Instances minorityInstances = new Instances(data);
        Instances majorityInstances = new Instances(data);
        separateDecisionClasses(data, minorityInstances, majorityInstances);

        // Pokaż statystyki przykladow
        showInitialDataStatitistics(data, minorityInstances, majorityInstances);


        XMeans xm = new XMeans();
        xm.setOptions(weka.core.Utils.splitOptions("weka.clusterers.XMeans -I 1 -M 1000 -J 1000 -L 2 -H 14 -B 1.0 -C 0.5 -D \"weka.core.EuclideanDistance -R first-last\" -S 10"));
        setClusterer(xm);
        // Usuń klasę decyzyjną -> uczenie nienadzorowane
        //TODO usun atrybuty nominalne lub je zamien na numeryczne
        NominalToBinary ntb = new NominalToBinary();
        ntb.setInputFormat(minorityInstances);
        Instances minorityInstancesNominalConvert = Filter.useFilter(minorityInstances, ntb);
        Instances majorityInstancesNominalConvert = Filter.useFilter(majorityInstances, ntb);
        Instances dataNominalConvert = Filter.useFilter(data, ntb);
        weka.filters.unsupervised.attribute.Remove filter = new weka.filters.unsupervised.attribute.Remove();
        filter.setAttributeIndices("" + (majorityInstancesNominalConvert.classIndex() + 1));
        filter.setInputFormat(minorityInstancesNominalConvert);
        Instances minorityInstancesNoClass = Filter.useFilter(minorityInstancesNominalConvert, filter);
        Instances majorityInstancesNoClass = Filter.useFilter(majorityInstancesNominalConvert, filter);
        Instances dataNoClass = Filter.useFilter(dataNominalConvert, filter);


        // Stwórz skupiska
        String options;
        //if (autoParametrizationEnable) {
            //options = getParametersForKMeans(minorityInstancesNoClass, filename);
           // clusterer.setOptions(weka.core.Utils.splitOptions(options));
       // }

        clusterer.buildClusterer(minorityInstancesNoClass);

        // Pokaż wyniki analizy skupisk
        ClusterEvaluation evaluation = new ClusterEvaluation();
        evaluation.setClusterer(clusterer);
        evaluation.evaluateClusterer(minorityInstancesNoClass);
        System.out.println(evaluation.clusterResultsToString());

        // Klasyfikuj przyklady klasy wiekszosciowej do skupisk (nie modyfukuje skupisk)
        Map<Integer, Integer> majorityHistogram = createHistogram(clusterer, majorityInstancesNoClass);

        Map<Integer, Integer> minorityHistogram = createHistogram(clusterer, minorityInstancesNoClass);

        // Pokaz statystyki
        showHistogram(majorityHistogram, "Majority");
        showHistogram(minorityHistogram, "Minority");
        showCombinedHistograms(majorityHistogram, minorityHistogram);

        // Rozdziel przyklady z klasami decyzyjnymi wedlug wyznaczonych skupisk
        Map<Integer, Instances> clustersAssignmentsMap = createClusterAssignmentsMap(data, dataNoClass, clusterer);
        showClusterAssignmentsMap(clustersAssignmentsMap);

        //SMOTE + Klasyfikacja
        System.out.println("Classifiers");
        for (Map.Entry<Integer, Instances> entry : clustersAssignmentsMap.entrySet()) {


            FilteredClassifier fc = new FilteredClassifier();
            if (smoteEnable) {
                System.out.println("SMOTE filtering");
                SMOTE smote = new SMOTE();
                System.out.println(getSMOTEPercentage(entry.getValue()));
                smote.setOptions(weka.core.Utils.splitOptions("-C 0 -K 5 -P " + getSMOTEPercentage(entry.getValue()) + " -S 1"));
                smote.setInputFormat(entry.getValue());
                fc.setFilter(smote);
            }
            fc.setClassifier(classifier);
            Evaluation eval = new Evaluation(entry.getValue());
            eval.crossValidateModel(fc, entry.getValue(), 10, new Random(1));

            System.out.println(eval.toSummaryString("\nResults\n======\n", false));
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());

        }


    }

    @Override
    public Instances loadDataFile(String filename) {
        this.filename = filename;
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
