import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn import datasets

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np 

from scipy.io import loadmat # needed to load in mat files

#############################################################################################################
# DOCUMENTATION
#############################################################################################################

# AUTHOR: Sabrina
# DATE CREATED: 02-23-2024
# FUNCTION: A suite of visualizations for the Indian Pines dataset, and PCA dimensionality reduction.
# PURPOSE: TBD

#############################################################################################################  

# Requirements for the assignment, listed here for my reference

# CS488/588 Homework 3 -70 pts
# 1A. Write a python program for dimensionality reduction using PCA on the Iris and Indian Pines dataset that implements the following: (25 points)
# i) For PCA, plot the explained variance for all the PC’s in the dataset. (5 points)
# ii) Reduce data visualization using PCA to 2 dimensions – display the new transformed data which is reduced to two dimensions for visualization. – display the first two PC’s (directions of projections) with respect to color-coded class separability.
# (10 points: 5 points per dataset for PCA plots, i.e. 5pts per plots)
# iii) Reduced data visualization using LDA to 2 dimensions – display the new transformed data which is reduced to two dimensions for visualization. – display the first two directions in LDA (directions of projections) with respect to color-coded class separability.
# (10 points: 5 points per dataset for LDA plots, i.e. 5pts per plots)
# *Note only provide data visualizations here.
# The number of dimensions chosen for analysis from i) can be anything – provide your justification in 1b for your choice. For visualization in ii) and iii) only plot the first two of dimensions chosen. Also, number of PCs or LDs chosen must be same for all analysis.

# For the sake of not wanting to go insane, I will be splitting the code into two parts: one for the Indian Pines dataset (Part A), and one for the Iris dataset (Part B).

#############################################################################################################
# GLOBALS
#############################################################################################################

# set the path to the desktop
desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

# find the paths to the datasets
indian_pines_path = os.path.join(desktop, 'indianR.mat')  # Dataset path
indian_pines_gt_path = os.path.join(desktop, 'indian_gth.mat')  # Ground truth path

# load the indian pines data
df = loadmat(indian_pines_path)

gth_mat = loadmat(indian_pines_gt_path)
gth_mat = {i : j for i, j in gth_mat.items() if i[0] != '_'}
gt = pd.DataFrame({i : pd.Series(j[0]) for i, j in gth_mat.items()})

# set our dataframes for the dataset and ground truth to more readable names
x_pines = df['X']
gth_pines = gt['gth']

# the iris dataset is built into sklearn, so we can just load it in. handy!
iris = datasets.load_iris()
x_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
gth_iris = pd.DataFrame(data=iris.target, columns=['target'])

#############################################################################################################
# FUNCTIONS
#############################################################################################################

# plot the explained variance for all the principal components, given the explained variance and a title
def plot_explained_variance(ev, title):
    plt.bar([i for i in range(1, len(ev) + 1)], list(ev), label = 'Principal Components', color = 'aqua')
    plt.legend()

    plt.xlabel('Principal Components')
    plt.ylabel('Variance Ratio')
    plt.title(title)

    plt.show()

# dimensionality reduction via PCA, returns the ev, principal components, and the dataframe
def pca_reduction(data, gth, desiredComponents):
    # apply PCA to the dataset
    pca = PCA(n_components = desiredComponents)
    principalComponents = pca.fit_transform(data)

    # explained variance ratio
    ev = pca.explained_variance_ratio_

    # convert the principal components to a dataframe
    principalDf = pd.DataFrame(data = principalComponents, columns = ['PC-' + str(i) for i in range(1, desiredComponents + 1)])

    # concatenate principal components with the ground truth labels
    finalDf = pd.concat([principalDf, gth], axis = 1)

    return ev, principalComponents, finalDf

# reduce the data to the desired number of components using LDA
def lda_reduction(data, gth, desiredComponents):
    # initialize the LDA model
    lda = LDA(n_components=desiredComponents)
    
    # fit the model and transform the dataset
    linearDiscriminants = lda.fit_transform(data,gth)
    
    # create a DataFrame for the reduced components
    columns = ['LD-' + str(i) for i in range(1, desiredComponents + 1)]
    principalDf = pd.DataFrame(data=linearDiscriminants, columns=columns)
    
    # concatenate the linear discriminants with the ground truth labels
    finalDf = pd.concat([principalDf, gth], axis=1)
    
    return linearDiscriminants, finalDf

# a function that simplifies the math for dimensionality reduction
def reduce_dimensions(data, desiredComponents):
    x1 = data.T
    X_pca = np.matmul(x1, desiredComponents)

    return X_pca

# a function that compares the classification results of different classifiers and returns the results
def compare_classifications(X, y, training_sizes, classifiers):
    results = []
    for size in training_sizes:
        size_results = {'Training Size': size, 'Classifiers': {}}
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=42)
        
        # loop through each classifier
        for name, clf in zip(classifier_names, classifiers):
            # train the classifier
            clf.fit(X_train, y_train)
            
            # predict on training and testing sets
            y_train_pred = clf.predict(X_train)
            y_test_pred = clf.predict(X_test)
            
            # calculate accuracies
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # store accuracies in the results
            size_results['Classifiers'][name] = {'Training Accuracy': train_accuracy, 'Testing Accuracy': test_accuracy}
            print(f"Classifier: {name}, Training Size: {size}, Training Accuracy: {train_accuracy}, Testing Accuracy: {test_accuracy}")
        
        # append the results for this training size to the main results list
        results.append(size_results)
    return results


def plot_comparison_results(results, training_sizes, classifier_names, titleString = "", fileString = "", display = True):
    plt.figure(figsize=(12, 12))
    
    # loop through each classifier to plot their performance
    for name in classifier_names:
        # prepare lists to store the accuracies for plotting
        train_accuracies = []
        test_accuracies = []

        # extract the accuracies for the current classifier across all training sizes
        for result in results:
            classifier_result = result['Classifiers'][name]
            train_accuracies.append(classifier_result['Training Accuracy'])
            test_accuracies.append(classifier_result['Testing Accuracy'])

        # plot the training and testing accuracies for the classifier
        plt.plot(training_sizes, train_accuracies, marker='o', linestyle='-', label=f'{name} - Training Acc')
        plt.plot(training_sizes, test_accuracies, marker='s', linestyle='--', label=f'{name} - Testing Acc')

    # adding graph title and labels
    plt.title('Classifier Performance Across Different Training Sizes ' + titleString)
    plt.xlabel('Training Size Fraction')
    plt.ylabel('Accuracy')

    # displaying the legend to differentiate between lines
    plt.legend()

    # show grid for better readability
    plt.grid(True)
    
    # save the plot with an appropriate name
    # start with the string "VIS_" to denote that it is a visualization, then the extra string
    
    plt.savefig(f'VIS_Classifier_Performance_{fileString}.png')

    if display:
        plt.show()

    
# as much as id love to write a function to plot the data, there are too many variables to account for.

#############################################################################################################
# BEGIN PCA ON INDIAN PINES DATASET (PART A)
#############################################################################################################

# set the number of desired components for PCA
desiredComponents = 10

# normalize the data between 0 and 1
scaler_model = MinMaxScaler()
scaler_model.fit(x_pines.astype(float))
x = scaler_model.transform(x_pines) 

# apply PCA to the dataset using the function
ev, principalComponents, X_pca_df = pca_reduction(x, gth_pines, desiredComponents)

# dimensionality reduction via PCA 
X_pca = reduce_dimensions(x, principalComponents)

# model the dataframe 
x_pca_df = pd.DataFrame(data = X_pca, columns = ['PC-' + str(i) for i in range(1, desiredComponents + 1)])

# add the labels 
X_pca_df = pd.concat([x_pca_df, gt], axis = 1)

# print number of features and classes
print(f'Number of features: {x_pines.shape[1]}')
print(f'Number of classes: {len(np.unique(gth_pines))}')

# i.A) For PCA, plot the explained variance for all the PC’s in the dataset. (5 points)
plt.bar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], list(ev), label = 'Principal Components for Indian Pines', color = 'aqua')
plt.legend()

plt.xlabel('Principal Components')
plt.ylabel('Variance Ratio')

# ii.B) Reduce data visualization using PCA to 2 dimensions and display the first two PC’s with respect to color-coded class separability.

plt.figure(figsize=(10, 10))
ax = plt.subplot(1, 1, 1)
markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']

for target, marker in zip(np.unique(gth_pines), markers):
    indicesToKeep = X_pca_df['gth'] == target
    ax.scatter(X_pca_df.loc[indicesToKeep, 'PC-1'], X_pca_df.loc[indicesToKeep, 'PC-2'], s=10, marker=marker, label=target)

ax.set_xlabel('PC-1', fontsize=15)
ax.set_ylabel('PC-2', fontsize=15)
ax.set_title('2 Component PCA for Indian Pines', fontsize=20)
ax.legend(np.unique(gth_pines))
ax.grid()
plt.show()

#############################################################################################################
# BEGIN PCA ON IRIS DATASET (PART B)
#############################################################################################################

# for the iris dataset, the number of features is 4, so we will reduce the dataset to 2 dimensions
desiredComponents = 2

# normalize the data between 0 and 1
scaler_model = MinMaxScaler()
scaler_model.fit(x_iris.astype(float))
iris_df = scaler_model.transform(x_iris)

# apply PCA to the dataset using the function
ev, principalComponents, finalDf = pca_reduction(iris_df, gth_iris, desiredComponents)

# convert the principal components to a dataframe
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC-' + str(i) for i in range(1, desiredComponents + 1)])

# concatenate principal components with the ground truth labels
finalDf_iris = pd.concat([principalDf, gth_iris], axis = 1)

# i.B) For PCA, plot the explained variance for all the PC’s in the dataset. (5 points)
plt.bar([1, 2], list(ev), label = 'Principal Components for Iris', color = 'aqua')
plt.legend()

plt.xlabel('Principal Components')
plt.ylabel('Variance Ratio')


# ii.B) Reduce data visualization using PCA to 2 dimensions and display the first two PC’s with respect to color-coded class separability. (Iris dataset)
# Since the iris dataset is way simpler than Indian pines, we can just plot the data directly.

plt.figure(figsize=(10, 10))
ax = plt.subplot(1, 1, 1)
markers = ['o', 'v', '^']

for target, marker in zip(np.unique(gth_iris), markers):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PC-1'], finalDf.loc[indicesToKeep, 'PC-2'], s=10, marker=marker, label=target)
    
ax.set_xlabel('PC-1', fontsize=15)
ax.set_ylabel('PC-2', fontsize=15)
ax.set_title('2 Component PCA for Iris', fontsize=20)
ax.legend(np.unique(gth_iris))
ax.grid()
plt.show()

# collect both dataframes for the PCA-reduced datasets. We plan on doing classification with these later.
# first, collect our reduced data for the iris dataset
reduced_iris_PCA = finalDf_iris 
# then, for our reduced indian pines dataset (note to self, keep an eye on this data. there could be issues.)
reduced_indian_pines_PCA = X_pca_df


# iii.A) Reduced data visualization using LDA to 2 dimensions – display the new transformed data which is reduced to two dimensions for visualization. – display the first two directions in LDA (directions of projections) with respect to color-coded class separability.
# okay, this should be a similar process to PCA, but with LDA instead.
# we will use the same number of desired components as PCA for consistency.

#############################################################################################################
# BEGIN LDA ON INDIAN PINES DATASET (PART A)
#############################################################################################################

desiredComponents = 10

# normalize the data between 0 and 1
scaler_model = MinMaxScaler()
scaler_model.fit(x_pines.astype(float))
x = scaler_model.transform(x_pines) 

# apply LDA to the dataset using the function
linearDiscriminants, finalDf = lda_reduction(x.T, gth_pines, desiredComponents)

# plot the first two directions in LDA with respect to color-coded class separability
plt.figure(figsize=(10, 10))
ax = plt.subplot(1, 1, 1)
markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']

for target, marker in zip(np.unique(gth_pines), markers):
    indicesToKeep = finalDf['gth'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'LD-1'], finalDf.loc[indicesToKeep, 'LD-2'], s=10, marker=marker, label=target)
    
ax.set_xlabel('LD-1', fontsize=15)
ax.set_ylabel('LD-2', fontsize=15)
ax.set_title('2 Component LDA for Indian Pines', fontsize=20)
ax.legend(np.unique(gth_pines))
ax.grid()
plt.show()

# save the reduced dataframe for the LDA-reduced datasets for classification later
reduced_indian_pines_LDA = finalDf

#############################################################################################################
# BEGIN LDA ON IRIS DATASET (PART B)
#############################################################################################################

# for the iris dataset, the number of features is 4, so we will reduce the dataset to 2 dimensions
desiredComponents = 2

# normalize the data between 0 and 1
scaler_model = MinMaxScaler()
scaler_model.fit(x_iris.astype(float))
iris_df = scaler_model.transform(x_iris)


# apply LDA to the dataset using the function
linearDiscriminants, finalDf = lda_reduction(iris_df, gth_iris, desiredComponents)


# plot the first two directions in LDA with respect to color-coded class separability
plt.figure(figsize=(10, 10))
ax = plt.subplot(1, 1, 1)
markers = ['o', 'v', '^']

for target, marker in zip(np.unique(gth_iris), markers):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'LD-1'], finalDf.loc[indicesToKeep, 'LD-2'], s=10, marker=marker, label=target)
    
ax.set_xlabel('LD-1', fontsize=15)
ax.set_ylabel('LD-2', fontsize=15)
ax.set_title('2 Component LDA for Iris', fontsize=20)
ax.legend(np.unique(gth_iris))
ax.grid()
plt.show()

# save the reduced dataframe for the LDA-reduced datasets for classification later
reduced_iris_LDA = finalDf

print('Done! Now onto classification.')

#############################################################################################################
# CLASSIFICATION
#############################################################################################################

# Requirements for the assignment, listed here for my reference:

# 2a. Write a python program to perform supervised classification on the Iris and Indian Pines datasets using Naïve Bayes, and Support vector machines (with RBF and Poly kernel) classifiers for training sizes ={10%, 20%, 30%, 40%, 50%} for each of the below cases:
# i) with dimensionality reduction – Reduce data based on your choice of ‘K’ dimensions from 1a) using each of the dimensionality reduction methods (PCA, LDA) followed by supervised classification by the listed classifiers.
# ii) without dimensionality reduction – data is followed by supervised classification using the listed classifiers.
# iii) Provide the plots for overall training accuracy, and overall classification accuracy vs. the training size for all methods (classification schemes). Tabulate the classwise classification accuracies (i.e. extension of the sensitivity and specificity values) only for 30% training size over all methods for each dataset for case i) i.e. with dimensionality reduction PCA and LDA for Indian pines dataset only.
# (30 pts: Total 4 plots, i.e. 2 plots for each dataset [case i and ii, overall training, and testing accuracy] + 2 tables (classwise accuracy), i.e, 1 table per (PCA, LDA) = 30 pts total, i.e. 5 pts per plots/table)

# remember the reduced dataframes we saved earlier? we will use those for classification.
# variable references:
# reduced_iris_PCA, reduced_indian_pines_PCA, reduced_iris_LDA, reduced_indian_pines_LDA

#############################################################################################################
# CLASSIFICATION GLOBALS
#############################################################################################################

# optional "mode" parameter so i dont have to wait for the program to train unnecessary models
# the option "all" will run classification for all datasets and reduction methods
mode = "all"

# set to true to display the plots, will be saved regardless
display = False

# set the training sizes, 10% to 50%
training_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
# set the classifiers
classifier_names = ['Naive Bayes', 'SVM with RBF', 'SVM with Poly']
classifiers = [GaussianNB(), SVC(kernel='rbf'), SVC(kernel='poly')]

#############################################################################################################
# CLASSIFICATION ON PINES DATASET (PCA)
#############################################################################################################

if mode == 1 or mode == "all":


    # prepare the data for classification
    X = reduced_indian_pines_PCA.iloc[:, :-1]  # features
    y = reduced_indian_pines_PCA.iloc[:, -1]   # labels

    print("Starting classification on Indian Pines dataset with PCA...")

    results = compare_classifications(X, y, training_sizes, classifiers)
    plot_comparison_results(results, training_sizes, classifier_names, "On Pines Data Using PCA", "INDIAN_PINES_PCA", display)

if mode == 2 or mode == "all":
    #############################################################################################################
    # CLASSIFICATION ON PINES DATASET (LDA)
    #############################################################################################################

    X = reduced_indian_pines_LDA.iloc[:, :-1]  # features
    y = reduced_indian_pines_LDA.iloc[:, -1]   # labels

    print("Starting classification on Indian Pines dataset with LDA...")

    results = compare_classifications(X, y, training_sizes, classifiers)
    plot_comparison_results(results, training_sizes, classifier_names, "On Pines Data Using LDA", "INDIAN_PINES_LDA" , display)

if mode == 3 or mode == "all":
    #############################################################################################################
    # CLASSIFICATION ON IRIS DATASET (PCA)
    #############################################################################################################

    X = reduced_iris_PCA.iloc[:, :-1]   # features
    y = reduced_iris_PCA.iloc[:, -1]    # labels
    
    print("Starting classification on Iris dataset with PCA...")
    
    results = compare_classifications(X, y, training_sizes, classifiers)
    plot_comparison_results(results, training_sizes, classifier_names, "On Iris Data Using PCA", "IRIS_PCA" , display)

    
if mode == 4 or mode == "all":
    #############################################################################################################
    # CLASSIFICATION ON IRIS DATASET (LDA)
    #############################################################################################################

    X = reduced_iris_LDA.iloc[:, :-1]  # features
    y = reduced_iris_LDA.iloc[:, -1]   # labels
    
    print("Starting classification on Iris dataset with LDA...")
    
    results = compare_classifications(X, y, training_sizes, classifiers)
    plot_comparison_results(results, training_sizes, classifier_names, "On Iris Data Using LDA", "IRIS_LDA", display)
    
# Now for the unreduced datasets

if mode == 5 or mode == "all":
    #############################################################################################################
    # CLASSIFICATION ON PINES DATASET (UNREDUCED)
    #############################################################################################################

    X = x_pines
    y = gth_pines

    print("Starting classification on Indian Pines dataset without dimensionality reduction...")

    results = compare_classifications(X.T, y, training_sizes, classifiers)
    plot_comparison_results(results, training_sizes, classifier_names, "On Pines Data Without Dimensionality Reduction", "INDIAN_PINES_UNREDUCED" , display)

if mode == 6 or mode == "all":
    #############################################################################################################
    # CLASSIFICATION ON IRIS DATASET (UNREDUCED)
    #############################################################################################################

    X = iris_df
    y = gth_iris

    print("Starting classification on Iris dataset without dimensionality reduction...")

    results = compare_classifications(X, y, training_sizes, classifiers)
    plot_comparison_results(results, training_sizes, classifier_names, "On Iris Data Without Dimensionality Reduction", "IRIS_UNREDUCED" , False)