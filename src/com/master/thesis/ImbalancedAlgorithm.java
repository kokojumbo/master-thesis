package com.master.thesis;

import weka.clusterers.AbstractClusterer;
import weka.core.Instances;

/**
 * Created by Marcin Gumkowski on 23.03.14.
 */
public interface ImbalancedAlgorithm {

    public void start() throws Exception;

    public Instances loadDataFile(String filename);

    public double getMinorityValue(Instances instances) throws Exception;

    public void setClusterer(AbstractClusterer clusteringAlgorithm);

    public void setClustererOptions(String[] options);

    public void setFilters();


}
