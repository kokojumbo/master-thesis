package com.master.thesis;

import weka.core.Instances;

/**
 * Created by Marcin Gumkowski on 23.03.14.
 */
public interface ImbalancedAlgorithm {


    public void setParameters();

    public void getParameters();

    public void start() throws Exception;

    public Instances loadDataFile(String filename);

    public  double getMinorityValue(Instances instances) throws Exception;


}
