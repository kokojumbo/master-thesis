package com.master.thesis;

import com.master.thesis.utils.LoadUtils;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.experiment.RandomSplitResultProducer;
import weka.filters.supervised.instance.SMOTE;


import java.util.Random;

import static com.master.thesis.utils.ImbalancedUtils.getSMOTEPercentage;

/**
 * Created by Marcin Gumkowski on 18.08.14.
 */
public class Main {



    public static void main(String[] args) {
        Instances instances = LoadUtils.loadDataFile("paw02a-600-5-0-BI-ver-3-zmieszane.arff");
        Random rand = new Random(1);
        instances.randomize(rand);
        RandomSplitResultProducer  x = new RandomSplitResultProducer();

        if (instances.classIndex() == -1) {
            instances.setClassIndex(instances.numAttributes() - 1);

        }
        double percent = 66;
        int trainSize = (int) Math.round(instances.numInstances() * percent
                / 100);
        int testSize = instances.numInstances() - trainSize;
        Instances train = new Instances(instances, 0, trainSize);
        Instances test = new Instances(instances, trainSize, testSize);
         train.randomize(rand);
        test.randomize(rand);
        try {
            J48 classifier = new J48();
            classifier.setOptions(weka.core.Utils.splitOptions("-U -M 2"));


            FilteredClassifier fc = new FilteredClassifier();

//                SMOTE smote = new SMOTE();
//                smote.setOptions(weka.core.Utils.splitOptions("-C 0 -K 5 -P " + getSMOTEPercentage(train) + " -S 1"));
//                smote.setInputFormat(train);
//                fc.setFilter(smote);

            fc.setClassifier(classifier);
            fc.buildClassifier(train);
            Evaluation eval = new Evaluation(train);

            eval.evaluateModel(classifier, test);

            System.out.println(eval.toSummaryString());
            System.out.println(eval.toMatrixString());

        } catch (Exception e) {
            e.printStackTrace();
        }

    }
    }
