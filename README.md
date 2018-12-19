# Credit Card Fraud Detection Model  |  Logistic Regression

![The Creative Few](http://thecreativefew.com/assets/images/creative_few_logo.svg)

Keywords: Deep Learning, Neural Networks, Tensorflow, Logistic Regression, Credit Card Fraud Detection, Python, PyCharm IDE

## Things Need to Build/Replicate the Model
Download the dataset from Kaggle Inc. @ https://www.kaggle.com/mlg-ulb/creditcardfraud <br />
Weka (Windows, OS X, Linux) can be downloaded @ https://www.cs.waikato.ac.nz/ml/weka/downloading.html
`Iteration5.py` is the current version of the model. <br />

---

## Introduction
This model was developed as a senior project build, implementing Deep Learning, Tensorflow, Logistic Regression, in Python.

## The Model
The model in Overview:
  - Data Pre-processes
    - Read in the datasets (70/30 split train to test respectively) 
    - Randomize/Shuffle the Datasets
    - One-Hot Encode the Class Column
    - Normalize the Data (values between 0 - 1)
    - Split the X/y Values for both Train & Test Sets
    - Convert the Datasets into a Numpy Array
    - Calculate Legit to Fraud Ratio (optional)

  - Building the Computational Graph
    - Define the Shape/Dimensions of the Input and Output Tensor (input = 30, output = 2)
    - Define the Neurons within the 2 Hidden Layers (Num of Layers & Neurons per is Dependent on the User)
    - Define the Input Nodes for Training
    - Define the Input Nodes for Testing
    
  - Build the Neural Network (input, hidden layer 1, hidden layer 2, output)
    - Define Hidden Layer 1
    - Define Hidden Layer 2
    - Define Output Layer
    - Create Network Function
    - Create Prediction Variables
    - Define Cross Entropy (calcs the loss or distance comparing y_actual to y_predicted)
    - Define the Optimizer (Adam optimizer is used @ a rate of 0.005 within the model)
    - Define the Accuracy Calculator function
    - Define the Number of Epochs (training iterations)
    
 - Start Session/Run Session
   - Initialize all Variables
   - Run the Session Feeding in X_train and y_train
   - Calculate Cross Entropy (using training values)
   - Display Elapsed Time & Accuracy Output @ every 5th Epoch
 
 - Display Overall Metrics
   - Calc & Display Accuracy Score
   - Calc & Display Precision Score
   - Calc & Display Recall Score
   - Calc & Display F-1 Measure Score
   - Display Confusion Matrix

---

## Noteworthy Inclusions
Accuracy - Overall, how often is the model correct?<br />
Accuracy Rate = ( TP + TN ) / ( TP + FN + FP + TN )<br />
Where, TP = True Positive, TN = True Negative, FP = False Positive, and FN = False Negative 

Precision - When the model predicts “Fraud," how often is it correct?<br />
Precision = TP / ( TP + FP )<br />
Where, TP = True Positive and FP = False Positive

Recall - When it is “Fraud,” how often does it predict as such?<br />
Recall = TP / ( TP + FN )<br />
Where, TP = True Positive, and FN = False Negative

F-1 Measure - The F1 Measure score represents an average of the precision score and recall score.<br />
F-1 = 2 * ( Precision * Recall ) / Precision 
 
 **Confision Matrix**
 
![Confusion Matrix](http://thecreativefew.com/assets/images/matrix.svg)
<br />
<br />
Visualizing the Deep Neural Network.
 - Input Layer w/30 Features
 - Hidden Layer 1 w/300 Neurons
 - Hidden Layer 2 w/450 Neurons
 - Output Layer w/2 Classifications

 **Neural Network**
 
![Deep Neural Network w/2 Hidden Layers](http://thecreativefew.com/assets/images/neural_network.svg)

---

## Work To Be Done
The model is in flux and there is much room for improvement. <br />
Works that still need to be completed includes the implementation of:
  - 10 Fold Cross Validation
  - Feature Ranking
  - ROC
  - Expand Algorithm to Include:
    - KNN
    - Random Forest
    - Decision Tree
    - Naive Bayes

---

## Contact
Adam Garza
agarza@thecreativefew.com

---
License
----

MIT
