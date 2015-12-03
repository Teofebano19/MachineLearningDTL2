package mydtl;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;
import weka.classifiers.Classifier;
import weka.core.Capabilities.Capability;
import weka.core.*;

/**
 *
 * @author Teofebano, Andrey
 */
public class myID3 extends Classifier{
    // Attribute
    myID3[] child;
    private Attribute attrSeparator;
    private Attribute classAttribute;
    private double[] result;
    private double classValue;
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        // capabilities check
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
        for (int i=0;i<data.numClasses();i++){
            if (classCounter.elementAt(i)>0){
                entropy -= (classCounter.elementAt(i) * Utils.log2(classCounter.elementAt(i)));
            }
        }
        entropy /= (double) data.numInstances();
        return entropy + Utils.log2(data.numInstances());
    }
    
    private double computeIG(Instances data, Attribute attr){
       double IG = computeEntropy(data);
       Instances[] instances = split(data,attr);
       for (int i=0;i<instances.length;i++){
           if (instances[i].numInstances() != 0){
               IG -= ((double)instances[i].numInstances() / (double) data.numInstances()) * computeEntropy(instances[i]);
           }
       }
       return IG;
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
        if(trainingData.numInstances()!=0){
            double[] informationGains = new double[trainingData.numAttributes()];   Enumeration attEnum = trainingData.enumerateAttributes();
            while (attEnum.hasMoreElements()) {
                Attribute att = (Attribute) attEnum.nextElement();
                informationGains[att.index()] = computeIG(trainingData, att);
            }
            attrSeparator = trainingData.attribute(Utils.maxIndex(informationGains));


            if (Utils.eq(informationGains[attrSeparator.index()], 0)) {
                attrSeparator = null;
                result = new double[trainingData.numClasses()];
                Enumeration instEnum = trainingData.enumerateInstances();
                while (instEnum.hasMoreElements()) {
                    Instance inst = (Instance) instEnum.nextElement();
                    result[(int) inst.classValue()]++;
                }
                Utils.normalize(result);
                classValue = Utils.maxIndex(result);
                classAttribute = trainingData.classAttribute();
            } else {
                Instances[] splitData = split(trainingData, attrSeparator);
                child = new myID3[attrSeparator.numValues()];
                for (int i=0; i<attrSeparator.numValues(); i++) {
                    child[i] = new myID3();
                    child[i].buildTree(splitData[i]);
                }
            }
        }
    }
    
    @Override
    public double classifyInstance(Instance testingData) throws NoSupportForMissingValuesException, Exception{
        if (testingData.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("MyID3 can't handle such missing value");
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
          throw new NoSupportForMissingValuesException("MyID3 can't handle such missing value");
        }
        if (attrSeparator == null) {
            return result;
        } else { 
            return child[(int) instance.value(attrSeparator)].distributionForInstance(instance);
        }
    }
    
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);
        result.setMinimumNumberInstances(0);

        return result;
    }
}
