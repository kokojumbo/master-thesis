package com.master.thesis;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.clusterers.AbstractClusterer;
import weka.clusterers.XMeans;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.supervised.instance.SMOTE;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import static com.master.thesis.utils.ImbalancedUtils.createClusterAssignmentsMap;
import static com.master.thesis.utils.ImbalancedUtils.separateDecisionClasses;

/**
 * Created by Marcin Gumkowski on 19.06.14.
 */
public class ImbalancedClassifier implements Classifier, Serializable {


    private AbstractClusterer clusterer;

    Map<Integer, Classifier> classifiersAssigmentMap = new HashMap<Integer, Classifier>();

    private boolean smoteEnable = true;
    private Instances data;

    private void build(Instances data) throws Exception {

        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        this.data = data;
        // Rozdziel przykłady na dwie klasy: mniejszościową i większościową
        Instances minorityInstances = new Instances(data);
        Instances majorityInstances = new Instances(data);
        separateDecisionClasses(data, minorityInstances, majorityInstances);

        // Pokaż statystyki przykladow
        //showInitialDataStatitistics(data, minorityInstances, majorityInstances);


        XMeans xm = new XMeans();
        xm.setOptions(weka.core.Utils.splitOptions("weka.clusterers.XMeans -I 1 -M 1000 -J 1000 -L 2 -H 14 -B 1.0 -C 0.5 -D \"weka.core.EuclideanDistance -R first-last\" -S 10"));
        setClusterer(xm);
        // Usuń klasę decyzyjną -> uczenie nienadzorowane
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
        System.out.println("no " + clusterer.numberOfClusters());
        // Pokaż wyniki analizy skupisk
        // ClusterEvaluation evaluation = new ClusterEvaluation();
        //evaluation.setClusterer(clusterer);
        //evaluation.evaluateClusterer(minorityInstancesNoClass);
        //System.out.println(evaluation.clusterResultsToString());

        // Klasyfikuj przyklady klasy wiekszosciowej do skupisk (nie modyfukuje skupisk)
        // Map<Integer, Integer> majorityHistogram = createHistogram(clusterer, majorityInstancesNoClass);

        //Map<Integer, Integer> minorityHistogram = createHistogram(clusterer, minorityInstancesNoClass);

        // Pokaz statystyki
        //showHistogram(majorityHistogram, "Majority");
        //showHistogram(minorityHistogram, "Minority");
        //showCombinedHistograms(majorityHistogram, minorityHistogram);

        // Rozdziel przyklady z klasami decyzyjnymi wedlug wyznaczonych skupisk
        Map<Integer, Instances> clustersAssignmentsMap = createClusterAssignmentsMap(data, dataNoClass, clusterer);
        //showClusterAssignmentsMap(clustersAssignmentsMap);

        //SMOTE + Klasyfikacja
        //System.out.println("Classifiers");

        for (Map.Entry<Integer, Instances> entry : clustersAssignmentsMap.entrySet()) {


            FilteredClassifier fc = new FilteredClassifier();
            if (smoteEnable) {
                //System.out.println("SMOTE filtering");
                SMOTE smote = new SMOTE();
                //System.out.println(getSMOTEPercentage(entry.getValue()));
                smote.setOptions(weka.core.Utils.splitOptions("-C 0 -K 5 -P " + getSMOTEPercentage(entry.getValue()) + " -S 1"));
                smote.setInputFormat(entry.getValue());
                fc.setFilter(smote);
            }
            fc.setClassifier(AbstractClassifier.forName("weka.classifiers.trees.J48", weka.core.Utils.splitOptions("-U -M 2")));
            fc.buildClassifier(entry.getValue());
            classifiersAssigmentMap.put(entry.getKey(), fc);
//            Evaluation eval = new Evaluation(entry.getValue());
//            eval.crossValidateModel(fc, entry.getValue(), 10, new Random(1));
//
//            System.out.println(eval.toSummaryString("\nResults\n======\n", false));
//            System.out.println(eval.toClassDetailsString());
//            System.out.println(eval.toMatrixString());

        }


    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        build(data);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        Instance copyInstance = (Instance) instance.copy();
        Instances instances = new Instances(data);
        instances.delete();
        instances.add(copyInstance);
        if (instances.classIndex() == -1) {
            instances.setClassIndex(instances.numAttributes() - 1);
        }
        weka.filters.unsupervised.attribute.Remove filter = new weka.filters.unsupervised.attribute.Remove();
        filter.setAttributeIndices("" + (instances.classIndex() + 1));
        filter.setInputFormat(instances);
        instances = Filter.useFilter(instances, filter);

        int cluster = clusterer.clusterInstance(instances.firstInstance());
        Classifier classifier = classifiersAssigmentMap.get(cluster);
        return classifier.classifyInstance(instance);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        Instance copyInstance = (Instance) instance.copy();
        Instances instances = new Instances(data);
        instances.delete();
        instances.add(copyInstance);
        if (instances.classIndex() == -1) {
            instances.setClassIndex(instances.numAttributes() - 1);
        }
        weka.filters.unsupervised.attribute.Remove filter = new weka.filters.unsupervised.attribute.Remove();
        filter.setAttributeIndices("" + (instances.classIndex() + 1));
        filter.setInputFormat(instances);
        instances = Filter.useFilter(instances, filter);
        int cluster = clusterer.clusterInstance(instances.firstInstance());
        Classifier classifier = classifiersAssigmentMap.get(cluster);
        return classifier.distributionForInstance(instance);
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }


    public void setClusterer(AbstractClusterer clusterer) {
        this.clusterer = clusterer;
    }

    public void setSmoteEnable(boolean smoteEnable) {
        this.smoteEnable = smoteEnable;
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
        double result = (((1.0 * majorityInstances.numInstances() / minorityInstances.numInstances()) - 1) * 100.0);
        if (result >=0  ){
            return result;
        }
        return 0;


    }
}
