import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np 
from time import sleep

from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error

#############################################################################################################
# DOCUMENTATION
#############################################################################################################

# AUTHOR: Sabrina
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

#==============================================================================
# PART 1A: VISUALIZATIONS
#==============================================================================

# USEFUL VARIABLES
X = iris_dataframe # the matrix of features
y = iris.target # target values

names = iris.target_names # the names of the target values
iris_corr_matrix = iris_numerical.corr() # the correlation matrix of the iris dataset

# add the target column to the dataFrame
iris_dataframe['species'] = pd.Categorical.from_codes(y, names)
print(iris_dataframe.head())


# 1A_i: Display the correlation matrix of the iris dataset as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(iris_corr_matrix, annot=True, cmap='coolwarm')

# make the plot window not huge
plt.tight_layout()

plt.show()

# 1A_ii: load the iris dataset into a pair plot
sns.pairplot(iris_dataframe, hue="species", markers=["o", "s", "D"])
plt.show()

# clean up the plots
plt.close('all')
plt.clf()

#==============================================================================
# PART 2: LINEAR REGRESSION
#==============================================================================

# SETTINGS FOR THE LINEAR REGRESSION MODEL
determinism = None  # the random state for the train_test_split function
lifetimes = 25  # the number of times the model will be trained and tested

# PLOT SETTINGS
colors = ['b', 'g']  # colors for 80-20 and 20-80 split respectively
bar_width = 0.25  # width of the bars in the bar plot
index = np.arange(lifetimes)  # the label locations

# create the feature and target variables
X = iris_numerical.drop('petal length (cm)', axis=1)
y = iris_numerical['petal length (cm)']

# initialize the model
model = LinearRegression()

# initialize a list to store the RMSEs for each split and each generation
rmse_list = []

# run the model for each lifetime
for generation in range(lifetimes):
    
    for i, test_size in enumerate([0.2, 0.8]):
        
        # split the data and train the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=determinism)
        
        model.fit(X_train, y_train)
        # predict the test set
        y_pred = model.predict(X_test)
        
        # calculate the RMSE and add it to the list
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        
        rmse_list.append(rmse)
        
        # Use a random sample from X_test for prediction, ensuring it's not in the training set
        sample_index = X_test.sample(1, random_state=determinism).index[0]
        sample = X.loc[[sample_index]]
        predicted_length = model.predict(sample)
        
        print(f'Generation {generation+1} - {test_size*100}% Split:')
        print(f'Sample: {sample}')
        print(f'Predicted petal length: {predicted_length}')
        print(f'Actual petal length: {y.loc[sample_index]}')
        print(f'Intercept: {model.intercept_}')
        print(f'Slope: {model.coef_}')
        print(f'RMSE: {rmse}')
        print("\n")

# if you're curious, uncomment the below code block to see a side-by-side comparison of the RMSEs for each split configuration per generation.

# NOTE: this only really provides insight if lifetimes > 1 and the determinism is set to none, which results in different splits each time.
# the only reason this code is here is because I was curious to see if the model was consistently better with one split configuration over the other.

 # separate the RMSE values for each split configuration
rmse_80_20 = rmse_list[0::2]  # this is 80% of the data
rmse_20_80 = rmse_list[1::2]  # this is 20% of the data

# determine which split configuration is better on average
avg_rmse_80_20 = sum(rmse_80_20) / lifetimes
avg_rmse_20_80 = sum(rmse_20_80) / lifetimes

# print the more accurate split configuration 
if avg_rmse_80_20 < avg_rmse_20_80:
    print(f'On average, the 80% split configuration is better with an average RMSE of {avg_rmse_80_20}.')
else:
    print(f'On average, the 20% split configuration is better with an average RMSE of {avg_rmse_20_80}.')   


# if the model is ran for more than 25 generations, dont show the plot since it will be too large
if lifetimes > 25:
    print('The model was ran for more than 25 generations, so the bar plot will not be shown.')
    exit()

# generate the bars for each split configuration
bars1 = plt.bar(index, rmse_80_20, bar_width, color='b', label='80% split')
bars2 = plt.bar(index + bar_width, rmse_20_80, bar_width, color='g', label='20% split')

# add some text for labels, title, and custom x-axis tick labels
plt.xlabel('Generation')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('RMSE of Linear Regression Model by Generation')
plt.xticks(index + bar_width / 2, [f'Gen {i+1}' for i in range(lifetimes)], rotation=45)
plt.legend()

# show the plot
plt.tight_layout()
plt.show()