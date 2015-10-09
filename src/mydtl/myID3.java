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
        for (int i=0;i<data.numClasses();i++){
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
        
        // search for highest IG
        Vector<Double> listIG =  new Vector<Double>();
        int numAttribute = trainingData.numAttributes();
        listIG.setSize(numAttribute);
        for (int i=0;i<numAttribute;i++){
            listIG.setElementAt(Double.valueOf(0), i);
        }
        for (int i=0;i<numAttribute;i++){
            Attribute attr = trainingData.attribute(i);
            int attrIndex = attr.index();
            listIG.setElementAt(computeIG(trainingData, attr), attrIndex);
        }
        int indexMaxIG = listIG.indexOf(Collections.max(listIG));
        attrSeparator = trainingData.attribute(indexMaxIG);
        
        // set the root for the tree
        if (listIG.elementAt(indexMaxIG) == 0){ // leaf
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
            child = new myID3[size];
            for (int i=0;i<size;i++){
                child[i] = new myID3();
                child[i].buildTree(splittedData[i]);
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
        Capabilities C = super.getCapabilities();
        C.disableAll();

        // attributes
        C.enable(Capability.NOMINAL_ATTRIBUTES);

        // class
        C.enable(Capability.NOMINAL_CLASS);
        C.enable(Capability.MISSING_CLASS_VALUES);

        // instances
        C.setMinimumNumberInstances(0);

        return C;
    }
}
