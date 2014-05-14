package com.master.thesis;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Arrays;


public class Case1 {

    private double falsePositiveRate;

    private double truePositiveRateSick;

    private double precision;

    private double recall;

    private double fmeasure;

    private Instances instances;

    private String[] removedAttributes = {"TBG", "TBG measured", "goitre", "lithium", "query on thyroxine", "psych", "I131 treatment", "tumor", "query hyperthyroid", "on antithyroid medication"};

    private double correctPercentage = 0.0;

    private int k = 10;

    public Case1(String fileName, String classifierString, boolean removeAttributes, boolean doSmote) {

        try {
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(fileName);
            instances = new Instances(source.getDataSet());
            instances.setClassIndex(instances.numAttributes() - 1);
            instances.stratify(k);
            Remove remove = new Remove();
            int[] attributeIndices = new int[removedAttributes.length];
            for (int i = 0; i < removedAttributes.length; i++) {
                Attribute attribute = instances.attribute(removedAttributes[i]);
                attributeIndices[i] = attribute.index();
            }
            remove.setAttributeIndicesArray(attributeIndices);
            remove.setInputFormat(instances);
            if (removeAttributes) {
                instances = Filter.useFilter(instances, remove);
            }
            String[] classifierArray = classifierString.split(" ");
            String classifierClassName = classifierArray[0];
            String[] classifierArgsArray = Arrays.copyOfRange(classifierArray, 1, classifierArray.length);

            Classifier classifier = AbstractClassifier.forName(classifierClassName, classifierArgsArray);

            for (int i = 0; i < k; i++) {
                int firstIndex = (instances.numInstances() / k) * i;
                int lastIndex = firstIndex + (instances.numInstances() / k);
                Classifier newClassifier = AbstractClassifier.makeCopy(classifier);
                Instances test = new Instances(instances, firstIndex, instances.numInstances() / k);
                Instances train = new Instances(instances, 0, firstIndex);
                for (int j = lastIndex; j < instances.numInstances(); j++) {
                    train.add(instances.instance(j));
                }
                SMOTE smote = new SMOTE();
                smote.setInputFormat(train);
                if (doSmote) {
                    train = Filter.useFilter(train, smote);
                }
                newClassifier.buildClassifier(train);
                Evaluation evaluation = new Evaluation(train);
                evaluation.evaluateModel(newClassifier, test);
                //printFoldResults(evaluation);
                correctPercentage += evaluation.pctCorrect();
                falsePositiveRate += evaluation.falsePositiveRate(0);
                truePositiveRateSick += evaluation.truePositiveRate(1);
                precision += evaluation.weightedPrecision();
                recall += evaluation.weightedRecall();
                fmeasure += evaluation.weightedFMeasure();
                //System.out.println(evaluation.toSummaryString());
                //System.out.println(evaluation.toMatrixString());
            }
            correctPercentage /= k;
            falsePositiveRate /= k;

            truePositiveRateSick /= k;
            precision /= k;
            recall /= k;
            fmeasure /= k;
            //printAverageResults();
            printAverageResultsInLine();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void printFoldResults(Evaluation evaluation) {
        double[][] confusionMatrix = evaluation.confusionMatrix();
        System.out.println(" === Confusion Matrix ===");
        System.out.println("");
        System.out.println("a\t\tb\t\t<-- classified as");
        System.out.println(confusionMatrix[0][0] + "\t\t" + confusionMatrix[0][1] + "\t\ta = negative");
        System.out.println(confusionMatrix[1][0] + "\t\t" + confusionMatrix[1][1] + "\t\tb = sick");
        System.out.println("");
        System.out.println("");
    }

    private void printAverageResults() {
        System.out.println("=== Average Results ===");
        System.out.println("");
        System.out.println("Correct percentage \t\t" + correctPercentage);
        //System.out.println("False positive rate \t" + (falsePositiveRate * 100));
        System.out.println("TruePositiveRateSick \t" + (truePositiveRateSick * 100));
        System.out.println("Precision \t\t\t\t" + precision);
        System.out.println("Recall \t\t\t\t\t" + recall);
        System.out.println("Fmeasure \t\t\t\t" + fmeasure);
        System.out.println(correctPercentage + "\t" + (truePositiveRateSick * 100) + "\t" + precision + "\t" + recall + "\t" + fmeasure);
    }

    private void printAverageResultsInLine() {
        System.out.println((correctPercentage) + "\t" + (truePositiveRateSick * 100) + "\t" + precision + "\t" + recall + "\t" + fmeasure);
    }
}
