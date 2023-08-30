# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 18:35:11 2021

Mukhtar Turarbek
201497773

"""

import pandas as pd
import numpy as np
import random as rnd


def count_features(num_features, N_train, num_all_feature_choices, num_classes, X_train, Y_train):
    # Count number of occurence of feature values in the training dataset
    #   Inputs:
    #   num_features - number of features
    #   N_train - number of training data
    #   num_all_feature_choices - the list with the number of choices for each feature
    #   num_classes - number of classes
    #   X_train - training dataframe with features
    #   Y_train - training series with labels
    #
    #   Output:
    #   count - list with numpy arrays. Each numpy array is related to the feature x.
    #       The numpy contains the information on the number of occurence of the feature value j
    #       with the label k. Rows and columns are taken as choices and labels, repectively.
    
    count = [] 
    
    # Count number of occurence of feature value j of the feature x with the label k in the training dataset
    for x in range(num_features):   # Iterate over the number of features
        
        # Occurence of feature value j (rows) of the feature x with the label k (columns)
        count_feature = np.zeros((num_all_feature_choices[x],num_classes))
        
        for i in range(N_train):    # Iterate over the number of training data
            for j in range(num_all_feature_choices[x]): # Iterate over the number of feature values of the feature x
                
                # Check if the feature value of the feature x matches with the loop
                if X_train.iloc[i][feature_names[x]] == j: 
                    
                    for k in range(num_classes):    # Iterate over the number of labels
                         # Check if the label of the i datum matches with the loop
                        if Y_train.iloc[i] == k: 
                            
                            # Compute occurence of feature value j (rows) of the feature x with the label k (columns)
                            count_feature[j,k] = count_feature[j,k]+1
                            break
                    break
                
        # Combine numpy count_feature arrays to the common count list
        count.append(count_feature)
    
    return count



def calculate_likelihood(count, num_features, num_all_feature_choices, num_classes):
    # Calculate likelihood in the training dataset
    #   Inputs:
    #   count - list with numpy arrays. Each numpy array is related to the feature x.
    #       The numpy contains the information on the number of occurence of the feature value j
    #       with the label k. Rows and columns are taken as choices and labels, repectively.
    #   num_features - number of features
    #   num_all_feature_choices - the list with the number of choices for each feature
    #   num_classes - number of classes
    #
    #   Output:
    #   like_count - list with numpy arrays. Each numpy array is related to the feature x.
    #       The numpy contains the information on the likelihood of the feature value j
    #       with the label k. Rows and columns are taken as choices and labels, repectively.
    
    
    like_count = []
    
    # Compute likelihood of feature value j of the feature x with the label i in the training dataset
    for x in range(num_features):   # Iterate over the number of features
        
        # Likelihood of feature value j (rows) of the feature x with the label k (columns)
        like_count_feature = np.zeros((num_all_feature_choices[x],num_classes))
        
        # Sum of the feature value occurences for the given feature x and label i
        sum_j_Nijk = np.zeros((num_classes))
        
        for i in range(num_classes):  # Iterate over the number of labels
            for j in range(num_all_feature_choices[x]):   # Iterate over the number of feature values of the feature x
                
                # Computes sum of the feature value occurences for the given feature x and label i
                sum_j_Nijk[i] = sum_j_Nijk[i] + count[x][j,i]
        
        for i in range(num_classes):  # Iterate over the number of labels
            for j in range(num_all_feature_choices[x]):    # Iterate over the number of feature values of the feature x
                
                # Computes likelihood of feature value j (rows) of the feature x with the label k (columns)
                like_count_feature[j,i] = count[x][j,i]/sum_j_Nijk[i]
        
        # Combine numpy like_count_feature arrays to the common like_count list
        like_count.append(like_count_feature)
    
    
    
    return like_count



def calculate_prior(num_classes, N_train, Y_train): #df_train
    # Calculate prior from the training dataset
    #   Inputs:
    #   num_classes - number of classes
    #   X_train - training dataframe with features
    #   Y_train - training datafram with labels
    #
    #   Output:
    #   prior_class - numpy array with prior for each label z
    
    
    prior_class = []
    
    # Compute prior of the label z in the training dataset
    for z in range(num_classes):    # Iterate over the number of labels
        
        # Number of occurence of the label z in the training data
        count_class_i = 0.0
        
        for i in range(N_train):    # Iterate over the number of training data
            # Check if the label z matches with the training datum i
            if z == Y_train.iloc[i]: 
                # Compute number of occurence of the label z in the training data
                count_class_i = count_class_i+1
        
        # Calculate prior for the label z
        prior_class_i = count_class_i/N_train
        
        # Combine prior_class_i to the common prior_class list
        prior_class.append(prior_class_i)
    
    return prior_class



def calculate_weight(likelihood_features, prior_class, num_features, num_all_feature_choices, num_classes):
    # Calculate weights for KNNNB classifier from the training dataset
    #   Inputs:
    #   likelihood_features - list with numpy arrays. Each numpy array is related to the feature x.
    #       The numpy contains the information on the likelihood of the feature value j
    #       with the label k. Rows and columns are taken as choices and labels, repectively.
    #   prior_class - numpy array with prior for each label z
    #   num_features - number of features
    #   num_all_feature_choices - the list with the number of choices for each feature
    #   num_classes - number of classes
    #
    #   Output:
    #   weight - list with numpy arrays. Each numpy array is related to the feature x.
    #       The numpy contains the information on the weight of the feature value j
    #       with the label i. Rows and columns are taken as choices and labels, repectively.
    
    weights = []
    
    # Compute weights for KNNNB classifier from the training dataset
    for x in range(num_features):   # Iterate over the number of features
    
        # Weight of the feature value j (rows) and label i (columns) of the feature x 
        weight_x = np.zeros((num_all_feature_choices[x], num_classes))
        
        for i in range(num_classes):    # Iterate over the number of labels
            for j in range(num_all_feature_choices[x]): # Iterate over the number of feature values of the feature x
                # Compute weight of the feature value j (rows) and label i (columns) of the feature x 
                weight_x[j,i] = likelihood_features[x][j,i]*prior_class[i]
        
        # Combine weight_x to the common weights list
        weights.append(weight_x)
            
    return weights


def encode_col(col, all_choices, df2, N):
    # Encode string from the dataframe's column with integers
    #   Inputs:
    #   col - column name
    #   all_choices - possible string of a column
    #   df2 - dataframe with data
    #   N - number of data in the dataframe
    #
    #   Output:
    #   df_en - the dataframe with converted column's strings to integers
    
    # Convert column from the dataframe to list
    col_param = df2[col].tolist()
    
    M = len(all_choices) # Number of choices
    col_param_en = [] # List where encoded column values will be put
    df_en = df2
    
    # Encode column strings to integers
    for i in range(N): # Iterate over the number of data
        for j in range(M):  # Iterate over the number of choices
            # Check if a column's string matches to the choice j
            if col_param[i] == all_choices[j]:
                # Encod column value to integer
                col_param_en.append(j)
                break
    
    # Remove the old column with string values from the dataframe
    df_en = df_en.drop(col, axis=1)
    
    # Add new column with integers from the dataframe
    df_en[col] = col_param_en
    
    return df_en



def KNNNB_train(num_features, class_col, N_train, num_all_feature_choices, num_classes, X_train, Y_train):
    # Train KNNNB classifier from the training data
    #   Inputs:
    #   num_features - number of features
    #   class_col - name of the label class
    #   N_train - number of training data
    #   num_all_feature_choices - the list with the number of choices for each feature
    #   num_classes - number of classes
    #   X_train - training dataframe with features
    #   Y_train - training series with labels
    #
    #   Output:
    #   weights - list with numpy arrays. Each numpy array is related to the feature x.
    #       The numpy contains the information on the weight of the feature value j
    #       with the label i. Rows and columns are taken as choices and labels, repectively.
    
    
    # Count occurence of feature values
    count = count_features(num_features, N_train, num_all_feature_choices, num_classes, X_train, Y_train)
    
    # Calculate likelihood for features
    likelihood_features = calculate_likelihood(count, num_features, num_all_feature_choices, num_classes)
    
    # Calculate prior of labels
    prior_class= calculate_prior(num_classes, N_train, Y_train)
    
    # Calculate weights
    weights = calculate_weight(likelihood_features, prior_class, num_features, num_all_feature_choices, num_classes)
    
    return weights




def KNNNB_predict(pt_X_test, K, num_features, num_classes, weights, idxs_train, N_train, Y_train_dec, X_train, Y_train):
    # Predict class and value by KNNNB classifier
    #   Inputs:
    #   pt_X_test - a list with features
    #   K - number of neighbors
    #   num_features - number of features
    #   num_classes - number of classes
    #   weights - list with numpy arrays. Each numpy array is related to the feature x.
    #       The numpy contains the information on the weight of the feature value j
    #       with the label i. Rows and columns are taken as choices and labels, repectively.
    #   idxs_train - a list of training data's IDs
    #   N_train - number of training data
    #   Y_train_dec - decoded series of labels
    #   X_train - training dataframe with features
    #   Y_train - training series with labels
    #
    #   Output:
    #   pred_Y - predicted class by KNNNB classifier
    #   pred_val - predicted value by KNNNB classifier
    
    
    Ed = [] # List of Euclidean distance between pt_X_test and training data
    
    #  Calculate Euclidean distances between pt_X_test and training data
    for i in range(N_train):    # Iterate over number of training data
        
        # Convert training dataframe with features to list
        inp_i = X_train.loc[idxs_train[i], :].values.tolist()
        
        Ed_i = 0    #  Euclidean distance between pt_X_test and training datum
        
        for j in range(num_classes):    # Iterate over number of labels
            for k in range(num_features):   # Iterate over number of features
                # Compute distance between pt_X_test and training datum
                Ed_i = Ed_i + (weights[k][inp_i[k],j]-weights[k][pt_X_test[k],j])**2
        
        # Compute Euclidean distance between pt_X_test and training datum
        Ed_i = Ed_i**0.5
        # Combine Ed_i to the common Ed list
        Ed.append(Ed_i)
    
    # Convert Ed list to numpy array
    Ed = np.array(Ed)
    
    # Sort Ed in ascending order and take the values' indexes
    sorted_idxs = np.argsort(Ed)
    
    # K neigbors to the pt_X_test
    K_neigh = np.zeros((K), dtype = int)
    
    # Find K neigbors to the pt_X_test
    for i in range(K):  # Iterate over K neighbors
        # Take the first K neigbors to the pt_X_test from sorted_idxs
        K_neigh[i] = sorted_idxs[i]
    
    score_vect = np.zeros((num_classes)) # number of votes to label from K neighbors
    pred_val = 0
    
    # Compute vote of the K neighbors and predicted value value for the pt_X_test
    for i in range(K):  # Iterate over K neighbors
        
        # Find vote of the neighbor i
        idx = Y_train.iloc[K_neigh[i]]
        
        # Consider vote of the neighbor i
        score_vect[idx]= score_vect[idx]+1
        
        # Compute predicted value (Part 1)
        pred_val = pred_val + Y_train_dec.iloc[K_neigh[i]]
    
    # Compute predicted value (Part 2)
    pred_val = pred_val/K
    
    # Predict label from votes of the K neighbors
    pred_Y = np.argmax(score_vect)
    
    return pred_Y, pred_val



# Load data
df = pd.read_csv("myData_1.csv", sep=';')

class_col = 'max load (MW)' # Column of the class in the dataframe

# Converting outputs from the dataset to list
Y = df[class_col].tolist()


df2 = df # Creating the new dataframe, where Y will be changed to labels

# Drop outrput from df2
df2 = df2.drop([class_col], axis=1)

newY = []

N = len(Y) # number of samples

vl_l_boun = 51500 # boundary between "Very low" class and "low" class
l_h_boun = 63200 # boundary between "low" class and "high" class
h_vh_boun = 75000 # boundary between "low" class and "high" class

# Change output values to labels
for i in range(N):
    if Y[i]< vl_l_boun:
        newY.append('Very low')
    elif Y[i]< l_h_boun:
        newY.append('Low')
    elif Y[i]< h_vh_boun:
        newY.append('High')
    else:
        newY.append('Very high')

# Assign new output column to df2
df2[class_col] = newY





feature_names = ['season', 'time', 'holidays', 'events']

df2_en = df2
col = 'season'
all_choices = ['winter', 'spr/fall', 'summer']
all_classes = ['Very low', 'Low', 'High', 'Very high']

# Encode features
df2_en = encode_col(col, all_choices, df2_en, N)
df2_en = encode_col('time', ['T1', 'T2', 'T3'], df2_en, N)
df2_en = encode_col('holidays', ['Y', 'N'], df2_en, N)
df2_en = encode_col('events', ['Y', 'N'], df2_en, N)

# Encode labels
df2_en = encode_col(class_col, all_classes, df2_en, N)



N_train = int(0.8*N) # number of training samples
N_test = N-N_train # number of training samples


# Divinde samples between training and testing datasets
idxs = list(range(N))
rnd.seed(124)
rnd.shuffle(idxs)


num_all_feature_choices = [3,4,2,2] #number of choices for the given feature

num_classes = 4 # number of labels
num_features = len(feature_names)   # number of features
K = 3

folds = []  # a list of fold items
num_folds = 7   # number of folds
fold_sizes = [] # a list of fold sizes

fold_size_i = int(N/7)  # first fold sizes

# Allocate items to folds
for i in range(num_folds):  # Iterate over number of folds
    
    if i!=num_folds-1:
        fold_sizes.append(fold_size_i)
        folds.append(idxs[fold_size_i*i:fold_size_i*(i+1)])
    else:
        fold_sizes.append(N-fold_sizes[-1]*6)
        folds.append(idxs[N-fold_sizes[-1]:N])

p_error_list = []   # a list with probbility errors from cross-validation

# Cross-validation
for k in range(num_folds):  # Iterate over number of folds
    idxs_train = [] # Indexes for training
    
    # Allocate folds to training dataset
    for j in range(num_folds):  # Iterate over number of folds
        if k==j:
            continue
        idxs_train = idxs_train+folds[j]
    
    # Allocate fold to testing dataset
    idxs_test = folds[k]
    
    N_train = len(idxs_train)   # number of training data
    N_test = len(idxs_test)     # # number of testing data
    
    df_train = df2_en.take(idxs_train)  # training dataframe
    df_test = df2_en.take(idxs_test)    # testing dataframe
    
    
    X_train = df_train[feature_names]   # training features
    Y_train = df_train[class_col]       # training labels
    X_test = df_test[feature_names]     # testing features
    Y_test = df_test[class_col]         # testing labels
    
    
    Y_train_dec = df.take(idxs_train)[class_col]    # training ouyput values
    Y_test_dec = df.take(idxs_test)[class_col]      # testing output values
    
    # Train KNNNB classifier
    weights = KNNNB_train(num_features, class_col, N_train, num_all_feature_choices, num_classes, X_train, Y_train)
    
    Y_pred = np.zeros((N_test))     # predicted label
    pred_val = np.zeros((N_test))   # predicted output value
    
    # Compute predicted values and labels for the training dataset
    for i in range(N_test): # Iterate over number of testing data
        
        # Take features of the testing datum i
        pt_X_test = X_test.iloc[i, :].values.tolist()
        
        # Compute predicted value and label for the testing datum i
        Y_pred[i], pred_val[i] = KNNNB_predict(pt_X_test, K, num_features, num_classes, weights, idxs_train, N_train, Y_train_dec, X_train, Y_train)
    
    
    
    p_error = 0.0   # error probability
    error = 0       # error count
    
    # Error count estimation
    for i in range(N_test): # Iterate over number of testing data
        # Consider error between actual and predicted testing datum i
        if Y_pred[i] != Y_test.iloc[i]:
            error = error+1
    
    # Calculate error probability
    p_error = error/N_test
    
    # Attach p_error to p_error_list
    p_error_list.append(p_error)

# Compute expected error probability
expected_p_error = sum(p_error_list)/len(p_error_list)

# Print expected error probability
print("Expected error of KNNNB is ", expected_p_error)



