package com.master.thesis.core;

import weka.clusterers.AbstractClusterer;
import weka.core.Instances;

/**
 * Created by Marcin Gumkowski on 23.03.14.
 */
public interface ImbalancedAlgorithm {

    public void start() throws Exception;

    public void setClusterer(AbstractClusterer clusteringAlgorithm);

    public Instances loadDataFile(String filename);

    public void smoteEnable(boolean enable);

    public void setClassifier(String classifier, String[] classifierOptions);

    public void setAutoParameterization(boolean enable);
}
