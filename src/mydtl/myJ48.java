/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package mydtl;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Utils;

/**
 *
 * @author Teofebano
 */
public class myJ48 extends Classifier{
    // Attributes
    myJ48[] child;
    private Attribute attrSeparator;
    private double[] result;
    private double classValue;
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();
        buildTree(data);
    }
    
    private double computeEntropy(Instances data){
        double entropy = 0;
        // Class
        int numClasses = data.numClasses();
        Vector<Double> classCounter = new Vector<Double>();
        classCounter.setSize(numClasses);
        for (int i=0;i<numClasses;i++){
            classCounter.setElementAt(Double.valueOf(0), i);
        }

        int numInstances = data.numInstances();
        for (int i=0;i<numInstances;i++){
            int cv = (int) data.instance(i).classValue();
            classCounter.setElementAt(classCounter.elementAt(cv)+1, cv);
        }
        // Entropy calculation for each class
        for (int i=0;i<numClasses;i++){
            if (classCounter.elementAt(i)>0){
                double probability = classCounter.elementAt(i) / numInstances;
                entropy -= (probability * Utils.log2(probability));
            }
        }
        return entropy;
    }
    
    private double computeIG(Instances data, Attribute attr){
       double IG = computeEntropy(data);
       Instances[] instances = split(data,attr);
       for (int i=0;i<instances.length;i++){
           if (instances[i].numInstances() > 0){
               IG -= (instances[i].numInstances() / data.numInstances()) * computeEntropy(instances[i]);
           }
       }
       return IG;
    }
    
    private double computeSplitInfo(Instances data, Attribute attr){
        double SI = 0;
        Instances[] instances = split(data,attr);
        for (int i=0;i<instances.length;i++){
            int numSplitted = instances[i].numInstances();
            double probability = numSplitted / data.numInstances();
            SI += probability * Utils.log2(probability);
        }
        return (SI*-1);
    }
    
    private double computeGR(Instances data, Attribute attr){
        double GR = computeIG(data,attr);
        if (computeSplitInfo(data,attr)!=0){
            GR /= computeSplitInfo(data,attr);
        }
        return GR;
    }
    
    private Instances[] split(Instances data, Attribute attr){
        Instances[] splitData = new Instances[attr.numValues()];
        for (int j = 0; j < attr.numValues(); j++) {
          splitData[j] = new Instances(data, data.numInstances());
        }
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
          Instance inst = (Instance) instEnum.nextElement();
          splitData[(int) inst.value(attr)].add(inst);
        }
        for (int i = 0; i < splitData.length; i++) {
          splitData[i].compactify();
        }
        return splitData;
    }
    
    private void buildTree(Instances trainingData) {
        // zero instance
        if (trainingData.numInstances() == 0){
            attrSeparator = null;
            result = new double[trainingData.numClasses()];
            classValue = Instance.missingValue();
            return;
        }
        
        // search for highest GR
        int numAttribute = trainingData.numAttributes();
        double[] listGR =  new double[numAttribute];
        for (int i=0;i<numAttribute;i++){
            listGR[i]=0;
        }
        for (int i=0;i<numAttribute;i++){
            Attribute attr = trainingData.attribute(i);
            int attrIndex = attr.index();
            listGR[attrIndex]=computeGR(trainingData, attr);
        }
        int indexMaxGR = Utils.maxIndex(listGR);
        attrSeparator = trainingData.attribute(indexMaxGR);
        
        // set the root for the tree
        if (listGR[indexMaxGR] == 0){ // leaf
            attrSeparator = null;
            int numClasses = trainingData.numClasses();
            int numInstances = trainingData.numInstances();
            result = new double[numClasses];
            for (int i=0;i<numInstances;i++){
                result[(int)trainingData.instance(i).classValue()]++;
            }
            Utils.normalize(result);
            classValue = Utils.maxIndex(result);
        }
        else{ // branch
            Instances[] splittedData = split(trainingData,attrSeparator);
            int size = attrSeparator.numValues();
            child = new myJ48[size];
            for (int i=0;i<size;i++){
                child[i] = new myJ48();
                child[i].buildTree(splittedData[i]);
            }
        }
    }
    
    @Override
    public double classifyInstance(Instance testingData) throws NoSupportForMissingValuesException, Exception{
        if (testingData.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("MyJ48 can't handle such missing value");
        }
        if (attrSeparator == null){
            return classValue;
        }
        else{
            return child[(int) testingData.value(attrSeparator)].classifyInstance(testingData);
        }
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws NoSupportForMissingValuesException {
        if (instance.hasMissingValue()) {
          throw new NoSupportForMissingValuesException("MyJ48 can't handle such missing value");
        }
        if (attrSeparator == null) {
            return result;
        } else { 
            return child[(int) instance.value(attrSeparator)].distributionForInstance(instance);
        }
    }
    
    @Override
    public Capabilities getCapabilities() {
        Capabilities C = super.getCapabilities();
        C.disableAll();

        // attributes
        C.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);

        // class
        C.enable(Capabilities.Capability.NOMINAL_CLASS);
        C.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        C.setMinimumNumberInstances(0);

        return C;
    }
}
