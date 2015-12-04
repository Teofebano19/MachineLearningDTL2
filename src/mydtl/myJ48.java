package mydtl;

import java.util.Enumeration;
import java.util.Vector;
import java.util.stream.DoubleStream;
import weka.classifiers.Classifier;
import weka.core.*;

/**
 *
 * @author Teofebano, Andrey
 */
public class myJ48 extends Classifier{
    // Attributes
    myJ48[] children;
    private Attribute attrSeparator;
    private Attribute classAttribute;
    private double[] result;
    private double classValue;
    private boolean isLeaf;
    private double threshold = 0;
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();
        
        buildTree(data);
        //pruneTree(data);
    } 
    
    // TREE
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
       Instances[] instances;
       if (attr.isNumeric()){
           instances = splitNumeric(data,attr);
       }
       else{
           instances = split(data,attr);
       }
       for (int i=0;i<instances.length;i++){
           if (instances[i].numInstances() != 0){
               IG -= ((double)instances[i].numInstances() / (double) data.numInstances()) * computeEntropy(instances[i]);
           }
       }
       return IG;
    }
    
    private double computeSplitInfo(Instances data, Attribute attr){
        double SI = 0;
        Instances[] instances;
        if (attr.isNumeric()){
            instances = splitNumeric(data,attr);
        }
        else{
            instances = split(data,attr);
        }
        for (int i=0;i<instances.length;i++){
            int numSplitted = instances[i].numInstances();
            if (numSplitted!=0){
                double probability = (double) (numSplitted / data.numInstances());
                SI -= probability * Utils.log2(probability);
            }
        }
        return SI;
    }
    
    private double computeGR(Instances data, Attribute attr){
        double GR = computeIG(data,attr);
        if (computeSplitInfo(data,attr)!=0){
            GR = (double) GR / computeSplitInfo(data,attr);
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
    
    private Instances[] splitNumeric(Instances data, Attribute attr){
        Instances[] splitData = new Instances[2];
        double threshold = countThreshold(data,attr);
        
        for (int j = 0; j < 2; j++) {
          splitData[j] = new Instances(data, data.numInstances());
        }
        splitData[0].add(data.instance(0));
        Enumeration instEnum = data.enumerateInstances();
        instEnum.nextElement();
        while (instEnum.hasMoreElements()) {
          Instance inst = (Instance) instEnum.nextElement();
          if (inst.value(attr)<threshold){
            splitData[0].add(inst);
          }
          else{
            splitData[1].add(inst);
          }
        }
        for (int i = 0; i < splitData.length; i++) {
          splitData[i].compactify();
        }
        return splitData;
    }
    
    private void buildTree(Instances trainingData) {
        isLeaf = false;

        if (trainingData.numInstances() != 0){
            // search for highest GR
            double[] gainRatio = new double[trainingData.numAttributes()];   
            Enumeration attEnum = trainingData.enumerateAttributes();
            while (attEnum.hasMoreElements()) {
                Attribute att = (Attribute) attEnum.nextElement();
                gainRatio[att.index()] = computeGR(trainingData, att);
            }
            attrSeparator = trainingData.attribute(Utils.maxIndex(gainRatio));
            
            if (gainRatio[attrSeparator.index()] == 0) { // Leaf
                attrSeparator = null;
                isLeaf = true;
                result = new double[trainingData.numClasses()];
                Enumeration instEnum = trainingData.enumerateInstances();
                while (instEnum.hasMoreElements()) {
                    Instance inst = (Instance) instEnum.nextElement();
                    result[(int) inst.classValue()]++;
                }
                Utils.normalize(result);
                classValue = Utils.maxIndex(result);
                classAttribute = trainingData.classAttribute();
            } else { // Branch
                if (isMissing(trainingData, attrSeparator)){
                    int index = findModusIndex(trainingData, attrSeparator);
                    
                    Enumeration instanceenum = trainingData.enumerateInstances();
                    while(instanceenum.hasMoreElements()){
                        Instance inst = (Instance) instanceenum.nextElement();
                        if(inst.isMissing(attrSeparator)){
                            inst.setValue(attrSeparator, attrSeparator.value(index));
                        }
                    }
                }
                
                Instances[] splitData;
                int size;
                if (attrSeparator.isNumeric()){
                    splitData = splitNumeric(trainingData,attrSeparator);
                    size = 2;
                    threshold = countThreshold(trainingData,attrSeparator);
                }
                else{            
                    splitData = split(trainingData,attrSeparator);
                    size = attrSeparator.numValues();
                }
                children = new myJ48[size];
                for (int i=0; i<size; i++) {
                    children[i] = new myJ48();
                    children[i].buildTree(splitData[i]);
                }
            }           
        }
    }
    
    private double pruneTree(Instances trainingData){
        double staticExpectedE = staticExpectedError((int) DoubleStream.of(result).sum(),
                (int) result[maxIndex(result)], result.length);
        
        if(isLeaf){
            return staticExpectedE;
        }else{
            double backUpE = 0;
            double totalInst = DoubleStream.of(result).sum();
            
            for(myJ48 childrenx : children){
                double totalChildrenInst = DoubleStream.of(childrenx.result).sum();
                backUpE = backUpE + totalChildrenInst / totalInst * childrenx.pruneTree(trainingData);
            }
            
            if(staticExpectedE < backUpE){
                attrSeparator = null;
                classValue = maxIndex(result);
                isLeaf = true;
                children = null;
                
                return staticExpectedE;
            }else{
                return backUpE;
            }
        }
    }
    
    // NUMERIC TO NOMINAL
    public double countThreshold(Instances trainingData, Attribute attr){
        double threshold = Double.MIN_VALUE;
        Vector<Double> numericValue = new Vector<Double>();
        for (int j=0;j<trainingData.numInstances();j++){
            numericValue.add(trainingData.instance(j).value(attr));
        }
        sort(numericValue);
        boolean done = false;
        for (int i=0;i<trainingData.numInstances()-1 && !done;i++){
            if (trainingData.instance(i) != trainingData.instance(i+1)){
                done = true;
                threshold = (double)(trainingData.instance(i).value(attr) + trainingData.instance(i+1).value(attr))/2;
            }
        }
        return threshold;
    }
    
    public void sort(Vector<Double> listValue){
        Double temp;
        
        for (int i=0;i<listValue.size()-1;i++){
            for (int j=1;j<listValue.size()-i;j++){
                if (listValue.elementAt(j-1)>listValue.elementAt(j)){
                    temp = listValue.elementAt(j-1);
                    listValue.setElementAt(listValue.elementAt(j), j-1);
                    listValue.setElementAt(temp, j);
                }
            }
        }
    }
    
    @Override
    public double classifyInstance(Instance testingData){
        int av;
        
        if (attrSeparator == null){
            return classValue;
        }
        else{
            if (attrSeparator.isNumeric()){
                if (testingData.value(attrSeparator)<threshold){
                    av = 0;
                }
                else{
                    av = 1;
                }
                return children[av].classifyInstance(testingData);
            }
            else{
                return children[(int) testingData.value(attrSeparator)].classifyInstance(testingData);
            }
        }
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) {
        int av;
        if (attrSeparator == null) {
            return result;
        } else {
            if (attrSeparator.isNumeric()){
                if (instance.value(attrSeparator)<threshold){
                    av = 0;
                }
                else{
                    av = 1;
                }
                return children[av].distributionForInstance(instance);
            }
            else{
                return children[(int) instance.value(attrSeparator)].distributionForInstance(instance);
            }
        }
    }
    
    public boolean isMissing(Instances instance, Attribute attr){
        boolean ismisval = false;
        Enumeration instanceenum = instance.enumerateInstances();
        
        while(instanceenum.hasMoreElements() && !ismisval){
            Instance inst = (Instance) instanceenum.nextElement();
            if(inst.isMissing(attr)){
                ismisval = true;
            }
        }
        
        return ismisval;
    }
    
    public int findModusIndex(Instances instance, Attribute attr){
        int modus[] = new int[attr.numValues()];
        Enumeration instanceenum = instance.enumerateInstances();
        
        while(instanceenum.hasMoreElements()){
            Instance inst = (Instance) instanceenum.nextElement();
            if(!inst.isMissing(attr)){
                modus[(int) inst.value(attr)]++;
            }
        }
        
        int max = 0;
        int index = -1;
        
        for(int i = 0; i < modus.length; i++){
            if(modus[i]>max){
                max = modus[i];
                index = i;
            }
        }
        
        return index;
    }
    
    public double staticExpectedError(int N, int n, int k){
        double E;
        
        E = (N - n + k - 1) / (double) (N + k);
        
        return E;
    }
    
    public double backedUpError(){
        double error = 0;
        double totalInst = DoubleStream.of(result).sum();
        
        for(myJ48 child:children){
            double totalChildInst = DoubleStream.of(child.result).sum();
            error = error + totalChildInst / totalInst 
                    * staticExpectedError((int) totalChildInst,
                            (int) child.result[(int)child.classValue],
                            child.result.length);
                
        }
        return error;
    }
    
    public int maxIndex(double[] array) {
        int max = 0;
        
        if (array.length > 0) {
            for (int i = 1; i < array.length; ++i) {
                if (array[i] > array[max]) {
                    max = i;
                }
            }
            return max;
        } else {
            return -1;
        }
    }
}
