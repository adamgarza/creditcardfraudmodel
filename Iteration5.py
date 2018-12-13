# ---------------------------------------------------------------------------------
# CREDIT CARD FRAUD DETECTION MODEL
# KEY WORDS: LOGISTIC REGRESSION / DEEP LEARNING / DEEP NEURAL NETWORK / TENSORFLOW / PYTHON / PYCHARM /
# ADAM GARZA
# ITS490 PROJECT
# FALL 2018
# ---------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import sklearn
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score


# ---------------------------------------------------------------------------------
# TABLE OF CONTENT #
# ---------------------------------------------------------------------------------
# DATA PREPROCESS
    # 1. READ IN THE DATASET
    # 2. RANDOMIZES/SHUFFLES THE DATASET
    # 3. ONE-HOT ENCODING
        # SPLITS THE CLASS COLUMN INTO [ 1 0 ] OR [ 0 1 ] FOR THE ONE-HOT VALUES
        # 0 TO --> [1 0] = LEGIT
        # 1 TO --> [0 1] = FRAUD
    # 4. NORMALIZE THE DATA
        # FORMATS THE VALUES TO BE WITHIN RANGE OF 0 & 1
        # SAME AS (x - mean(x)) / std(x)
    # 5. SEPARATING THE X & Y VALUES
        # X VALUES WILL BE ALL THE (FEATURES) COLUMNS V1, V2, ..., V28  - DROP() DROPS THE CLASSIFICATION COLUMNS
    # 6. CONVERT THESE DATA FRAMES INTO NUMPY ARRAYS
    # 7. SPLITTING THE DATA INTO 2 HALVES (TRAIN & TEST)
        # EACH HALF HAS 2 SECTIONS (X_TRAIN & Y_TRAIN) AND (X_TEST & Y_TEST)
        # 80/20 SPLIT - 80% TRAINING TO 20% TESTING
    # 8. CALCULATE RATIO FOR LOGIT VS FRAUD
        # DATA IS BIASED : HAS GREATER AMOUNT OF LEGITIMATE TRANSACTIONS
        # LOGIT WEIGHTs WILL BE IMPLEMENTED FOR ORIGINAL DATASET
    # 9. LOGIT WEIGHTS ARE APPLIED TO UNDERREPRESENTED CLASS

# BUILDING THE COMPUTATIONAL GRAPH COMPONENTS (LAYERS & NODES)
    # 10. DEFINING THE SHAPE OF X & y (TENSOR OF 30 & TENSOR OF 2 RESPECTIVELY)
    # 11. DEFINING NEURONS IN THE 2 HIDDEN LAYERS
    # 12. DEFINING INPUTS FOR TRAINING NODES
        # INPUT TENSORS TRAINING NODES - THESE WILL BE POPULATED WITH CONTENT WHILE PASSING THROUGH THE MODEL
        # WILL PASS RAW_X_TRAIN AND RAW_Y_TRAIN INTO THESE NODES
    # 13. DEFINING INPUTS FOR TESTING NODES
        # DURING TESTING, THESE WILL BE USED AS INPUTS INTO THE MODEL (TEST CASES)

# DEFINING LAYERS OF THE NEURAL NETWORK
    # THE OUTPUT OF ONE LAYER IS THE INPUT OF THE NEXT LAYER
    # THESE ARE VARIABLES BECAUSE THEY WILL CHANGE
    # GOAL IS TO CREATE A OPTIMIZED LOSS FUNCTION BY ADJUSTING THE VARIABLES THESE HAVE
    # 14. DEFINING HIDDEN LAYER 1
        # TAKES INPUT FROM INPUT_X (TENSOR OF 30) WITH 100 NEURONS AND PASSES THE OUTPUT TO HIDDEN LAYER 2
    # 15. DEFINING HIDDEN LAYER 2
        # TAKES IN INPUT FROM LAYER 1 AND PASSES THE OUTPUT TO LAYER 3
    # 16. DEFINING OUTPUT LAYER
        # TAKES IN THE INPUT OF LAYER 2 AND OUTPUTS [1 0] OR [0 1] DEPENDING ON CATEGORIZATION
    # 17. DEFINING NETWORK FUNCTION
        # THIS FUNCTION ACCEPTS AN INPUT TENSOR i.e., TRAINING NODES OR TESTING NODES
        # WILL MAKE A PREDICTION BY RUNNING IT THROUGH THIS LAYERED NETWORK
        # TF.NN = TENSORFLOW.NEURALNETWORKS - SIGMOID FUNCTION FITS THE DATA BEST
    # 18. CREATE VARIABLES TO HOLD THE PREDICTED VALUE
    # 19. CROSS ENTROPY - CALCULATES THE LOSS BY COMPARING THE ACTUAL OUTPUT TO THE PREDICTED OUTPUT
        # SOFTMAX IS USED BECAUSE ONE HOT MANIPULATION WAS USED
        # MATCH THE INDEX OF THE ACTUAL & PREDICTED
    # 20. ADAM OPTIMIZER IS USED TO MINIMIZE THE LOSS FUNCTION USING GRADIENT DESCENT (0.005 IS THE LEARNING RATE)
    # 21. COMPARES THE ACTUAL TO THE PREDICTED
        # RETURNS A PERCENTAGE BY MULTIPLYING THE RETURN VALUE BY 100
    # 22. DETERMINE VALUE FOR EPOCH ITERATIONS

# START SESSION - RUN THE MODEL
    # 23. INITIALIZE ALL VARIABLES
        # FOR LOOP IS USED TO ITERATE THROUGH THE N_EPOCHS COUNT
        # THE TIME() FUNCTION IS USED TO CAPTURE TIME ELAPSED DURING THE RUN OF INDIVIDUAL EPOCHS
        # I START THE SESSION AND STORE ITS RETURN VALUE INTO VARIABLE CROSS ENTROPY_SCORE
        # FEED DICTIONARY FEEDS IN 80% OF THE DATA SET PER SPLIT RATION IN SECTION 6 ABOVE
    # 24. RUN THE SESSION CALCULATING CROSS-ENTROPY GIVEN THE TRAINING SET
    # 25. DISPLAY OUTPUT @ EVERY 10TH EPOCH & CALCULATE THE TIME ELAPSED
    # 26. DISPLAY ACCURACY RATE @ EVERY 10TH EPOCH (Y_TEST_NODE VS Y_TEST_PREDICTION)

# METRICS OF THINGS
    # 27. ACCURACY SCORE CALCULATED
    # 28. PRECISION SCORE CALCULATED
    # 29. RECALL SCORE CALCULATED
    # 30. F1 MEASURE CALCULATED
    # 31. CONFUSION MATRIX CROSS TAB
# ---------------------------------------------------------------------------------


# START
# DATA PREPROCESSES ---------------------------------------------------------------
# 1. READ IN THE DATASET ----------------------------------------------------------
creditcard_dataset = pd.read_csv("creditcard_randomized.csv")   # SELECT DATASET FOR LOGIT, OVER SAMPLE OR UNDER SAMPLE
# IF YOU SELECT THE OVER SAMPLE OR UNDER SAMPLE, REMOVE THE LOGIT WEIGHTING PARAMETER


# 2. RANDOMIZES/SHUFFLES THE DATASET ----------------------------------------------
#randomized_dataset = creditcard_dataset.sample(frac=1)  # SHUFFLING THE DATASET USING SAMPLE.(FRACTION=0.8)
#print("Randomized Dataset:")
#print(randomized_dataset)
#print("------------------------------------------------------------------------")


# 3. ONE-HOT ENCODING -------------------------------------------------------------
one_hot_dataset = pd.get_dummies(creditcard_dataset, columns=["Class"])
print("one-hot: ")
print(one_hot_dataset)
print("------------------------------------------------------------------------")


# 4. NORMALIZE THE DATA -----------------------------------------------------------
norm_data = (one_hot_dataset - one_hot_dataset.min()) / (one_hot_dataset.max() - one_hot_dataset.min())
print("Normalized: (CONVERTS FEATURES & CLASSES INTO VALUES BETWEEN 0-1 EXCLUSIVELY)")
print(norm_data)
print("------------------------------------------------------------------------")


# 5. SEPARATING THE X & Y VALUES --------------------------------------------------
split_X = norm_data.drop(["Class_0", "Class_1"], axis=1)   # [ number of rows x 30 columns of features ]
print("split_X: (ALL FEATURES)")
print(split_X)
# Y VALUES WILL BE THE ONE_HOT_DATA (CLASSES) COLUMN
split_y = norm_data[["Class_0", "Class_1"]]    # [ number of rows x 2 columns of classes ]
print("split_y: (ALL CLASSIFICATIONS)")
print(split_y)
print("------------------------------------------------------------------------")


# 6. CONVERT THESE DATA FRAMES INTO NUMPY ARRAYS ----------------------------------
array_X = np.asanyarray(split_X.values, dtype="float32")
print("Array_X: (NUMPY ARRAY OF ALL FEATURES)")
print(array_X)
array_y = np.asanyarray(split_y.values, dtype="float32")
print("Array_y: (NUMPY ARRAY OF ALL CLASSIFICATIONS)")
print(array_y)
print("------------------------------------------------------------------------")


# 7. SPLITTING THE DATA INTO 2 HALVES (TRAIN & TEST) ------------------------------
train_size = int(0.8 * len(array_X))

# FOR TRAIN - GRAB EVERYTHING FROM 0 TO LENGTH TRAIN_SIZE (80% OF CONTENT)
raw_X_train = array_X[:train_size]
raw_y_train = array_y[:train_size]
#raw_X_train = random.shuffle(array_X[:train_size])
#raw_y_train = random.shuffle(array_y[:train_size])

print("raw_X_train :")
print(raw_X_train)
#print(len(raw_X_train))
print("raw_y_train: ")
print(raw_y_train)
print("\n")

# FOR TEST - GRAB EVERYTHING FORM TRAIN_SIZE TO THE END (20% OF CONTENT)
raw_X_test = array_X[train_size:]
raw_y_test = array_y[train_size:]
print("raw_X_test :")
print(raw_X_test)
#print(len(raw_X_test))
print("raw_y_test :")
print(raw_y_test)
print("------------------------------------------------------------------------")


# 8. CALCULATE RATIO FOR LEGIT VS FRAUD -------------------------------------------
# count_legit, count_fraud = np.unique(creditcard_dataset["Class"], return_counts=True)[1]
count_legit, count_fraud = np.unique(creditcard_dataset["Class"], return_counts=True)[1]
print("legit count", count_legit)
print("fraud count", count_fraud)
fraud_ratio = float(count_fraud / (count_legit + count_fraud))
legit_ratio = float(count_legit / (count_fraud + count_legit))
print("------------------------------------------------------------------------")
print("Percentage of Fraud: {0: .4f}%".format(100 * fraud_ratio))
print("Percentage of Legit: {0: .4f}%".format(100 * legit_ratio))
print("------------------------------------------------------------------------")


# 9. LOGIT WEIGHTS ARE APPLIED TO UNDERREPRESENTED CLASS --------------------------
# logit_weight = 1/fraud_ratio # PLACES GREATER IMPORTANCE ON UNDER-REPRESENTED CLASS BY INCREASING ITS WEIGHT
# raw_y_train[:, 1] = raw_y_train[:, 1] * logit_weight  # INCREASING THE WEIGHT OF ALL FRAUDULENT CLASSIFICATIONS
# print('\n Logit weight is: {0: .4f}'.format(logit_weight))
# print('\n raw_y_train is : \n', raw_y_train)



# BUILDING THE COMPUTATIONAL GRAPH COMPONENTS (LAYERS & NODES) --------------------
# ---------------------------------------------------------------------------------
# 10. DEFINING THE SHAPE OF X & y (TENSOR OF 30 & TENSOR OF 2 RESPECTIVELY) -------
input_X = array_X.shape[1]  # TENSOR OF 30
# ESTABLISHES THE SHAPE OF y - ONE-HOT ENCODED [0, 1] or [1, 0] (TENSOR OF 2 ELEMENTS)
output_y = array_y.shape[1]  # TENSOR OF 2
print("input_X: ", input_X)
print("output_y: ", output_y)
print("------------------------------------------------------------------------")


# 11. DEFINING NEURONS IN THE 2 HIDDEN LAYERS -------------------------------------
layer1_neurons = 300   # LAYER 1 WILL HAVE 300 CELLS
layer2_neurons = 450   # LAYER 2 WILL HAVE 450 CELLS


# 12. DEFINING INPUTS FOR TRAINING NODES ------------------------------------------
X_train_node = tf.placeholder(tf.float32, [None, input_X], name="X_train_node")  # INPUT TENSOR OF 30 (FEATURES)
y_train_node = tf.placeholder(tf.float32, [None, output_y], name="y_train_node") # OUTPUT TENSOR OF 2 (CLASSIFICATIONS)


# 13. DEFINING INPUTS FOR TESTING NODES -------------------------------------------
X_test_node = tf.constant(raw_X_test, name="X_test_node")    # TAKES IN THE RAW TEST INPUT
y_test_node = tf.constant(raw_y_test, name="y_test_node")    # TAKES IN THE RAW TEST OUTPUT



# LOGISTIC REGRESSION - DEFINING LAYERS OF THE NEURAL NETWORK ---------------------
# ---------------------------------------------------------------------------------
# 14. DEFINING HIDDEN LAYER 1 -----------------------------------------------------
hidden_layer1_node = tf.Variable(tf.random_normal([input_X, layer1_neurons]), name="layer1_weight_node")
hidden_layer1_biases_node = tf.Variable(tf.random_normal([layer1_neurons]), name="layer1_biases_node")


# 15. DEFINING HIDDEN LAYER 2 -----------------------------------------------------
hidden_layer2_node = tf.Variable(tf.random_normal([layer1_neurons, layer2_neurons]), name="layer2_weight_node")
hidden_layer2_biases_node = tf.Variable(tf.random_normal([layer2_neurons]), name="layer2_biases_node")


# 16. DEFINING OUTPUT LAYER 3 -----------------------------------------------------
output_node = tf.Variable(tf.random_normal([layer2_neurons, output_y]), name="layer3_weight")
output_biases_node = tf.Variable(tf.random_normal([output_y]), name="layer3_biases")


# 17. DEFINING NETWORK FUNCTION ---------------------------------------------------
# SEEN BELOW IN EACH LAYER --> TF.MATMUL((X * W) + B)
def network(input_tensor):
    layer1 = tf.nn.sigmoid(tf.matmul(input_tensor, hidden_layer1_node) + hidden_layer1_biases_node)
    layer2 = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(layer1, hidden_layer2_node) + hidden_layer2_biases_node), 0.85)
    output_prediction = tf.nn.softmax(tf.matmul(layer2, output_node) + output_biases_node)
    return output_prediction


# 18. CREATE VARIABLES TO HOLD THE PREDICTED VALUE --------------------------------
y_trian_prediction = network(X_train_node)
y_test_prediction = network(X_test_node)


# 19. CROSS ENTROPY - CALCULATES LOSS BY COMPARING ACTUAL TO THE PREDICTED - LOGISTIC REGRESSION ----------------------
xentropy = tf.losses.softmax_cross_entropy(y_train_node, y_trian_prediction)


# 20. ADAM OPTIMIZER IS USED TO MINIMIZE THE LOSS FUNCTION USING GRADIENT DESCENT (0.005 IS THE LEARNING RATE) --------
optimizer = tf.train.AdamOptimizer(0.005).minimize(xentropy)


# 21. COMPARES THE ACTUAL TO THE PREDICTED ----------------------------------------
def calc_accuracy(actual, predicted):
    actual = np.argmax(actual, 1)
    predicted = np.argmax(predicted, 1)
    return (100 * np.sum(np.equal(predicted, actual)) / predicted.shape[0])

# 22. DETERMINE VALUE FOR EPOCH ITERATIONS
n_epochs = 20



# START SESSION - RUN THE MODEL ---------------------------------------------------
# ---------------------------------------------------------------------------------
# 23. INITIALIZE ALL VARIABLES ----------------------------------------------------
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(n_epochs):
        start_time = time.time()

        # 24. RUN THE SESSION CALCULATING CROSS-ENTROPY GIVEN THE TRAINING SET
        _, xentropy_score = sess.run([optimizer, xentropy],
                                        feed_dict={X_train_node: raw_X_train, y_train_node: raw_y_train})

        # 25. DISPLAY OUTPUT @ EVERY 10TH EPOCH & CALCULATE THE TIME ELAPSED
        if epoch % 5 == 0:
            timer = time.time() - start_time
            print("Time elapsed: {0: .2f} seconds  |  ".format(timer),
                  "Current Loss: {0:.4f}  | ".format(xentropy_score), "@ Epoch: {}".format(epoch))

            # 26. DISPLAY ACCURACY RATE @ EVERY 10TH EPOCH (Y_TEST_NODE VS Y_TEST_PREDICTION)
            final_y_test = y_test_node.eval()
            final_y_test_prediction = y_test_prediction.eval()
            final_accuracy_per_epoch = calc_accuracy(final_y_test, final_y_test_prediction)
            print("Current Accuracy (At This Epoch Iteration): {0:.2f}%".format(final_accuracy_per_epoch))
            print("------------------------------------------------------------------------")

    # METRICS OF THINGS -----------------------------------------------------------
    # -----------------------------------------------------------------------------
    print("\n")
    print("Metrics of Fraud Detection Model | Logistic Regression:")
    print("---------------------------------")


    # 27. ACCURACY SCORE CALCULATED -----------------------------------------------
    actual = np.argmax(final_y_test, 1)
    predicted = np.argmax(final_y_test_prediction, 1)
    final_accuracy_overall = (100 * accuracy_score(actual, predicted))
    print("Final Accuracy: {0:.2f}%".format(final_accuracy_overall))


    # 28. PRECISION SCORE CALCULATED ----------------------------------------------
    precision = (100 * precision_score(y_true=actual, y_pred=predicted))
    print("Precision: {0:.2f}%".format(precision))


    # 29. RECALL SCORE CALCULATED -------------------------------------------------
    recall = (100 * recall_score(y_true=actual, y_pred=predicted))
    print("Recall: {0:.2f}%".format(recall))


    # 30. F1 MEASURE CALCULATED ---------------------------------------------------
    f_score = (100 * f1_score(y_true=actual, y_pred=predicted))
    print("F1 Measure: {0:.2f}%".format(f_score))
    print("\n")


    # 31. CONFUSION MATRIX CROSS TAB ----------------------------------------------
    print("Confusion Matrix Cross Tab: ")
    print("---------------------------------")
    print(pd.crosstab(actual, predicted, rownames=["True"], colnames=["Predicted"], margins=True))
    print("\n")
    print("---------------------------------- END ----------------------------------")
    print("\n")

