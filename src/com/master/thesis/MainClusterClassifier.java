package com.master.thesis;

import com.master.thesis.utils.LoadUtils;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.supervised.instance.SMOTE;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Random;

import static com.master.thesis.utils.ImbalancedUtils.getSMOTEPercentage;

/**
 * Created by Marcin Gumkowski on 26.06.14.
 */
public class MainClusterClassifier {

    public static void main(String[] args) throws FileNotFoundException {
        String[] fileNames = {"acl.arff", "new-thyroid.arff", "vehicle.arff", "ecoli.arff", "ionosphere.arff", "haberman.arff", "abalone.arff", "transfusion.arff", "yeast-ME2.arff", "paw02a-600-5-0-BI-ver-1-bez-zmian.arff","paw02a-600-5-0-BI-ver-2-oddalone-skupiska.arff","paw02a-600-5-0-BI-ver-3-zmieszane.arff","paw02a-600-5-0-BI-ver-4-outliery.arff"};

        PrintStream out = new PrintStream(new FileOutputStream("results\\NewCV.txt"));
        System.setOut(out);
        runnerCV(fileNames, false);
        idxMethod++;
        out = new PrintStream(new FileOutputStream("results\\NewCVSmote.txt"));
        System.setOut(out);
        runnerCV(fileNames, true);
        idxMethod++;
        out = new PrintStream(new FileOutputStream("results\\NewTT.txt"));
        System.setOut(out);
        runnerTrainTest(fileNames, false);
        idxMethod++;

        out = new PrintStream(new FileOutputStream("results\\NewTTSmote.txt"));
        System.setOut(out);
        runnerTrainTest(fileNames, true);
        idxMethod++;

        out = new PrintStream(new FileOutputStream("results\\TreeCV.txt"));
        System.setOut(out);
        runnerSimpleClasssifierCV(fileNames, false);
        idxMethod++;
        out = new PrintStream(new FileOutputStream("results\\TreeCVSmote.txt"));
        System.setOut(out);
        runnerSimpleClasssifierCV(fileNames, true);
        idxMethod++;
        out = new PrintStream(new FileOutputStream("results\\TreeTT.txt"));
        System.setOut(out);
        runnerSimpleClasssifierTrainTest(fileNames, false);
        idxMethod++;

        out = new PrintStream(new FileOutputStream("results\\TreeTTSmote.txt"));
        System.setOut(out);
        runnerSimpleClasssifierTrainTest(fileNames, true);

        out = new PrintStream(new FileOutputStream("results\\results.txt"));
        System.setOut(out);
        for (int k = 0; k < 13; k++) {

            System.out.println(fileNames[k]);
            for (int i = 0; i < 13; i++) {
                for (int j = 0; j < 8; j++) {
                    System.out.print(tab[j][k][i] + "\t");
                }
                System.out.println();
            }
            System.out.println();
            System.out.println();
        }
    }

    public static void runnerCV(String[] fileNames, boolean smoteEnable) {

        for (String filename : fileNames) {
            System.out.println("======================================================================");
            System.out.println("======================================================================");
            System.out.println("======================================================================");
            System.out.println("======================      " + filename + "       =======================");
            System.out.println("======================================================================");
            System.out.println("======================================================================");
            System.out.println("======================================================================");


            Instances instances = LoadUtils.loadDataFile(filename);

            if (instances.classIndex() == -1) {
                instances.setClassIndex(instances.numAttributes() - 1);

            }


            try {
                NominalToBinary ntb = new NominalToBinary();
                ntb.setInputFormat(instances);
                instances = Filter.useFilter(instances, ntb);

                ImbalancedClassifier classifier = new ImbalancedClassifier();
                classifier.setSmoteEnable(smoteEnable);
                classifier.buildClassifier(instances);

                Evaluation eval = new Evaluation(instances);
                eval.crossValidateModel(classifier, instances, 10, new Random(1));
                printResults(eval);

            } catch (Exception e) {
                e.printStackTrace();
            }


        }
    }

    public static void runnerTrainTest(String[] fileNames, boolean smoteEnable) {

        for (String filename : fileNames) {
            System.out.println("======================================================================");
            System.out.println("======================================================================");
            System.out.println("======================================================================");
            System.out.println("======================      " + filename + "       =======================");
            System.out.println("======================================================================");
            System.out.println("======================================================================");
            System.out.println("======================================================================");


            Instances instances = LoadUtils.loadDataFile(filename);
            instances.randomize(new Random(0));
            if (instances.classIndex() == -1) {
                instances.setClassIndex(instances.numAttributes() - 1);

            }
            double percent = 66.6;
            int trainSize = (int) Math.round(instances.numInstances() * percent
                    / 100);
            int testSize = instances.numInstances() - trainSize;
            Instances train = new Instances(instances, 0, trainSize);
            Instances test = new Instances(instances, trainSize, testSize);

            try {
                NominalToBinary ntb = new NominalToBinary();
                ntb.setInputFormat(train);
                train = Filter.useFilter(train, ntb);
                ntb.setInputFormat(test);
                test = Filter.useFilter(test, ntb);

                ImbalancedClassifier classifier = new ImbalancedClassifier();
                classifier.setSmoteEnable(smoteEnable);
                classifier.buildClassifier(train);

                Evaluation eval = new Evaluation(train);
                eval.evaluateModel(classifier, test);
                printResults(eval);

            } catch (Exception e) {
                e.printStackTrace();
            }


        }
    }

    public static void runnerSimpleClasssifierCV(String[] fileNames, boolean smoteEnable) {
        for (String filename : fileNames) {
            System.out.println("======================================================================");
            System.out.println("======================================================================");
            System.out.println("======================================================================");
            System.out.println("======================      " + filename + "       =======================");
            System.out.println("======================================================================");
            System.out.println("======================================================================");
            System.out.println("======================================================================");


            Instances instances = LoadUtils.loadDataFile(filename);

            if (instances.classIndex() == -1) {
                instances.setClassIndex(instances.numAttributes() - 1);

            }


            try {
                J48 classifier = new J48();
                classifier.setOptions(weka.core.Utils.splitOptions("-U -M 2"));


                FilteredClassifier fc = new FilteredClassifier();
                if (smoteEnable) {
                    SMOTE smote = new SMOTE();
                    smote.setOptions(weka.core.Utils.splitOptions("-C 0 -K 5 -P " + getSMOTEPercentage(instances) + " -S 1"));
                    smote.setInputFormat(instances);
                    fc.setFilter(smote);
                }
                fc.setClassifier(classifier);
                fc.buildClassifier(instances);
                Evaluation eval = new Evaluation(instances);
                eval.crossValidateModel(fc, instances, 10, new Random(1));
                printResults(eval);

            } catch (Exception e) {
                e.printStackTrace();
            }


        }

    }

    public static void runnerSimpleClasssifierTrainTest(String[] fileNames, boolean smoteEnable) {
        for (String filename : fileNames) {
            System.out.println("======================================================================");
            System.out.println("======================================================================");
            System.out.println("======================================================================");
            System.out.println("======================      " + filename + "       =======================");
            System.out.println("======================================================================");
            System.out.println("======================================================================");
            System.out.println("======================================================================");


            Instances instances = LoadUtils.loadDataFile(filename);
            instances.randomize(new Random(0));
            if (instances.classIndex() == -1) {
                instances.setClassIndex(instances.numAttributes() - 1);

            }
            double percent = 66.6;
            int trainSize = (int) Math.round(instances.numInstances() * percent
                    / 100);
            int testSize = instances.numInstances() - trainSize;
            Instances train = new Instances(instances, 0, trainSize);
            Instances test = new Instances(instances, trainSize, testSize);

            try {
                J48 classifier = new J48();
                classifier.setOptions(weka.core.Utils.splitOptions("-U -M 2"));


                FilteredClassifier fc = new FilteredClassifier();
                if (smoteEnable) {
                    SMOTE smote = new SMOTE();
                    smote.setOptions(weka.core.Utils.splitOptions("-C 0 -K 5 -P " + getSMOTEPercentage(train) + " -S 1"));
                    smote.setInputFormat(train);
                    fc.setFilter(smote);
                }
                fc.setClassifier(classifier);
                fc.buildClassifier(train);
                Evaluation eval = new Evaluation(train);

                eval.evaluateModel(fc, test);

                printResults(eval);


            } catch (Exception e) {
                e.printStackTrace();
            }


        }

    }


    public static String[][][] tab = new String[8][13][13];
    public static int idx = 0;
    public static int idxMethod = 0;

    public static void printResults(Evaluation eval) throws Exception {
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));

        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());

        //System.out.println("Sensitivity");
        System.out.println(round3Dec(eval.truePositiveRate(0)));
        tab[idxMethod][idx][0] = Double.toString(round3Dec(eval.truePositiveRate(0)));
        System.out.println(round3Dec(eval.truePositiveRate(1)));
        tab[idxMethod][idx][1] = Double.toString(round3Dec(eval.truePositiveRate(1)));
        System.out.println(round3Dec(eval.weightedTruePositiveRate()));
        tab[idxMethod][idx][2] = Double.toString(round3Dec(eval.weightedTruePositiveRate()));

        //System.out.println("Specificity");
        System.out.println(round3Dec(eval.trueNegativeRate(0)));
        tab[idxMethod][idx][3] = Double.toString(round3Dec(eval.trueNegativeRate(0)));
        System.out.println(round3Dec(eval.trueNegativeRate(1)));
        tab[idxMethod][idx][4] = Double.toString(round3Dec(eval.trueNegativeRate(1)));
        System.out.println(round3Dec(eval.weightedTrueNegativeRate()));
        tab[idxMethod][idx][5] = Double.toString(round3Dec(eval.weightedTrueNegativeRate()));

        //System.out.println("GMeans");
        System.out.println(round3Dec(Math.sqrt(eval.truePositiveRate(0) * eval.trueNegativeRate(1))));
        tab[idxMethod][idx][6] = Double.toString(round3Dec(Math.sqrt(eval.truePositiveRate(0) * eval.trueNegativeRate(0))));
        System.out.println(round3Dec(Math.sqrt(eval.truePositiveRate(1) * eval.trueNegativeRate(0))));
        tab[idxMethod][idx][7] = Double.toString(round3Dec(Math.sqrt(eval.truePositiveRate(1) * eval.trueNegativeRate(1))));
        System.out.println(round3Dec(Math.sqrt(eval.weightedTruePositiveRate() * eval.weightedTrueNegativeRate())));
        tab[idxMethod][idx][8] = Double.toString(round3Dec(Math.sqrt(eval.weightedTruePositiveRate() * eval.weightedTrueNegativeRate())));

        //System.out.println("FMeasure");
        System.out.println(round3Dec(eval.fMeasure(0)));
        tab[idxMethod][idx][9] = Double.toString(round3Dec(eval.fMeasure(0)));
        System.out.println(round3Dec(eval.fMeasure(1)));
        tab[idxMethod][idx][10] = Double.toString(round3Dec(eval.fMeasure(1)));
        System.out.println(round3Dec(eval.weightedFMeasure()));
        tab[idxMethod][idx][11] = Double.toString(round3Dec(eval.weightedFMeasure()));

        //System.out.println("Accuracy");
        System.out.println(round3Dec(eval.pctCorrect()) + " %");
        tab[idxMethod][idx][12] = round3Dec(eval.pctCorrect()) + " %";

        if (idx == 12) {
            idx = 0;
            System.out.println();
            System.out.println("Aggregated results");
            for (int i = 0; i < 13; i++) {
                for (int j = 0; j < 13; j++) {
                    // System.out.print(tab[j][i] + "\t");

                }
                System.out.println();
            }

        } else {
            idx++;
        }


    }

    public static double round3Dec(double value) {
        return (double) Math.round(value * 1000) / 1000;
    }

}

