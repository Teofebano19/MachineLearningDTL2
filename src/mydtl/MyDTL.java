package mydtl;

import java.io.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ConfusionMatrix;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Debug;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Teofebano, Andrey
 */
public class MyDTL {
    // Attribute
    private static final String SOURCE = "data/weather.nominal.arff";
    private static final int NUMBER_FOLD = 10;
    private static final int PERCENTAGE = 66;
    public static Instances data;
    
    // Load file
    public static void loadFile(String source){
        try {
            data = ConverterUtils.DataSource.read(source);
            if (data.classIndex() == -1){
                data.setClassIndex(data.numAttributes()-1);
            }
        } catch (Exception ex) {
            Logger.getLogger(MyDTL.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    // 10-fold
    public static void learn10fold(Instances trainingData, Classifier classifier){
        try {
            // Build and Eval
            Evaluation eval = new Evaluation(trainingData);
            eval.crossValidateModel(classifier, trainingData, NUMBER_FOLD, new Debug.Random(1));
            
            // Print
            System.out.println("=== Summary ===");
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
        } catch (Exception ex) {
            Logger.getLogger(MyDTL.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    // full
    public static void learnFull(Instances trainingData, Classifier classifier){
        try {
            // Build
            Classifier cls = classifier;
            int trainSize = trainingData.numInstances();
            Instances train = new Instances(trainingData, 0, trainSize);
            Instances test = new Instances(trainingData, 0, trainSize);
            cls.buildClassifier(train);
            // Eval
            Evaluation eval = new Evaluation(trainingData);
            eval.evaluateModel(classifier, test);
            
            // Print
            System.out.println("=== Summary ===");
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
        } catch (Exception ex) {
            Logger.getLogger(MyDTL.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    // percentage
    public static void learnPercentage(Instances trainingData, Classifier classifier){
        try {
            // Build
            Classifier cls = classifier;
            int trainSize = trainingData.numInstances();
            int threshold = trainSize * PERCENTAGE;
            Instances train = new Instances(trainingData, 0, threshold);
            Instances test = new Instances(trainingData, threshold + 1, trainSize);
            cls.buildClassifier(train);
            // Eval
            Evaluation eval = new Evaluation(trainingData);
            eval.evaluateModel(classifier, test);
            
            // Print
            System.out.println("=== Summary ===");
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
        } catch (Exception ex) {
            Logger.getLogger(MyDTL.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    // save model
    public static void saveModel(Instances trainingData, Classifier classifier, String file){
        try {
            Classifier cls = classifier;
            cls.buildClassifier(trainingData);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
                oos.writeObject(cls);
                oos.flush();
            }
        } catch (Exception ex) {
            Logger.getLogger(MyDTL.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    // load model
    public static void loadModel(String file){
        ObjectInputStream ois;
        try {
            Classifier cls = null;
            ois = new ObjectInputStream(new FileInputStream(file));
            cls = (Classifier) ois.readObject(); 
            ois.close();
        } catch (FileNotFoundException e){
        } catch (IOException | ClassNotFoundException ex) {
            Logger.getLogger(MyDTL.class.getName()).log(Level.SEVERE, null, ex);
        } 
    }
    
    // classification
    public static void classifyUsingModel(Classifier classifier, String file){
        try {
            Instances unlabeled = ConverterUtils.DataSource.read(file);
            unlabeled.setClassIndex(unlabeled.numAttributes() - 1);
            Instances labeled = new Instances(unlabeled);
            // label instances
            for (int i = 0; i < unlabeled.numInstances(); i++) {
                double clsLabel = classifier.classifyInstance(unlabeled.instance(i));
                labeled.instance(i).setClassValue(clsLabel);
                System.out.println(labeled.lastInstance().toString());
            }
        } catch (Exception ex) {
            Logger.getLogger(MyDTL.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    // attribute removal
    public static Instances removeAttribute(String file, String indices, Boolean invert ) throws Exception{
        Instances newdata;
        Remove remove = new Remove();
        remove.setAttributeIndices(indices);
        remove.setInvertSelection(invert.booleanValue());
        remove.setInputFormat(data);
        newdata = Filter.useFilter(data, remove);
        return newdata;
    }
    
    // filter resample
    public static Instances resample(Instances data) throws Exception{
        Instances newdata;
        Resample resample = new Resample();
        resample.setInputFormat(data);
        newdata = Filter.useFilter(data, resample);
        return newdata;
    }
    
    // main
    public static void main(String[] args) {
        String[] listData = new String[]{"data/weather.nominal.arff", "data/weather.numeric.arff", "data/iris.arff"};
        Classifier[] listClassifier = new Classifier[]{new myID3(), new Id3()};
        String[] listClassifierName = new String[]{"myID3", "Id3", "myJ48", "J48"};
        
        for (int i=0;i<listData.length;i++){
            for (int j=0;j<listClassifier.length;j++){
                System.out.println("-------------//---------------");
                System.out.println("Using " + listClassifierName[j] + " to classify " + listData[i]);
                
                loadFile(listData[i]);
                System.out.println("FULL");
                learnFull(data,listClassifier[j]);
                System.out.println(" ");
                System.out.println("10-FOLD");
                learn10fold(data,listClassifier[j]);
            }
        }
    }    
}
