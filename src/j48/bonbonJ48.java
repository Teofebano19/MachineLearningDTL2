/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package j48;
import j48.node;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Teofebano
 */
public class bonbonJ48 extends Classifier{
    // Attribute
    private node root = null;

    @Override
    public void buildClassifier(Instances trainingData) throws Exception {
        root = new node();
        root.buildClassifier(trainingData);
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return root.classifyInstance(instance);
    }
}
