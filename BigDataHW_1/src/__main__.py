import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import seaborn as sns
import numpy as np 
from sklearn.linear_model import LinearRegression

#############################################################################################################
# DOCUMENTATION
#############################################################################################################

# AUTHOR: Garrett Thrower
# DATE CREATED: 02-23-2024
# FUNCTION: A suite of visualizations for the Iris dataset, as well as an attempt at linear regression.
# PURPOSE: To analyze the Iris dataset and obtain meaningful insights from it.

#############################################################################################################  

# print the feature names and target names to check that the data was loaded correctly
# print(iris.feature_names)
# print(iris.target_names)

# the below comment is used for reference for myself when debugging
"""
    The output of the above print statements is as follows:
    
    ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    ['setosa' 'versicolor' 'virginica']
    
"""

# fetch the iris dataset with sklearn and load into a pandas dataframe
iris = datasets.load_iris()
iris_dataframe = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_numerical = iris_dataframe.select_dtypes(include=[np.number]) # the numerical columns of the iris dataset, important since we load the names of the target values into the dataframe later

# USEFUL VARIABLES
X = iris_dataframe # the matrix of features
y = iris.target # target values

names = iris.target_names # the names of the target values
iris_corr_matrix = iris_numerical.corr() # the correlation matrix of the iris dataset

# add the target column to the dataFrame
iris_dataframe['species'] = pd.Categorical.from_codes(y, names)


# 1A_i: Display the correlation matrix of the iris dataset as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(iris_corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# 1A_ii: load the iris dataset into a pair plot
sns.pairplot(iris_dataframe, hue="species", markers=["o", "s", "D"])
plt.show()

# TODO: WORK ON PART 2, LINEAR REGRESSION





