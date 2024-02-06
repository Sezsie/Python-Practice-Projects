# Linear Regression Calculations and Visualization
# AUTHOR: Garrett Thrower
# Last Modified: 2/3/2024

# This script calculates the coefficients for a linear regression model using the least squares method.
# All done manually, since the point is to understand the math behind it.
# It then plots the data and the line of best fit using matplotlib.

from time import sleep
import matplotlib.pyplot as plt

# parameters for the gradient descent, tweak them as you see fit

acceptable_error = 0.01 # the error threshold that will stop the gradient descent
learning_rate = 0.01 # how much the coefficients will be updated in each iteration
delay = 0.01 # delay between iterations in seconds

# this is the path to the file with the data on my computer, but probably not on yours. Change it to the path of your file
userPath = "C:/Users/User1/Desktop/data.txt" 


# this function will x and y data points from a file and return them as two separate lists
def extract_data(path):
    x_values = []
    y_values = []
    with open(path, 'r') as file:
        for line in file:
            if line.strip(): # This checks if the line is not empty
                x, y = map(float, line.split(','))
                x_values.append(x)
                y_values.append(y)
    return x_values, y_values


# this function will calculate the mean of a list of numbers
def mean(values, label):
    mean = sum(values) / len(values)
    print(f"Mean of {label} values: {mean}")
    return mean


# this function calculates theta_1 and theta_0 for the linear regression model's equation
def calculate_coefficients(x, y):
    x_mean = mean(x, "X")
    y_mean = mean(y, "Y")
    numerator = 0
    denominator = 0
    for i in range(len(x)):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)
        denominator += (x[i] - x_mean) ** 2
    theta_1 = numerator / denominator
    theta_0 = y_mean - theta_1 * x_mean
    return theta_1, theta_0



# plotting function
def plot_data(title, x, y, theta_1, theta_0):
    plt.scatter(x, y, color='blue', label='Data Points')
    # Calculate the y values of the line of best fit
    line_x = [min(x), max(x)]
    line_y = [theta_0 + theta_1 * x_value for x_value in line_x]
    plt.plot(line_x, line_y, color='red', label='Line of Best Fit')
    plt.xlabel('Profit in $10,000s')
    plt.ylabel('Population of City in 10,000s')
    plt.title(title)
    plt.legend()
    plt.show()


# batch gradient descent function, note that it will print the current error at each iteration   
def batch_gradient_descent(x, y, learning_rate=0.01, error_threshold=0.01, delay=0):
    m = len(y)
    theta_0 = 0
    theta_1 = 0
    previous_mse = float('inf')

    while True:
        sum_error_0 = 0
        sum_error_1 = 0
        mse = 0
        for i in range(m):
            prediction = theta_0 + theta_1 * x[i]
            error = prediction - y[i]
            sum_error_0 += error
            sum_error_1 += error * x[i]
            mse += error ** 2
        mse /= m
        print(f"CURRENT ERROR: {abs(previous_mse - mse)}")

        if abs(previous_mse - mse) < error_threshold:
            print(f"FINAL ERROR: {abs(previous_mse - mse)}")
            break

        previous_mse = mse  # Update the previous_mse for the next iteration
        theta_0 = theta_0 - (learning_rate * (1/m) * sum_error_0)
        theta_1 = theta_1 - (learning_rate * (1/m) * sum_error_1)
        sleep(delay)
        

    return theta_1, theta_0

# function-ified so I dont have to repeat the equation over and over again
def make_prediction(x, theta_1, theta_0):
    
    # this is basically just y = mx + b, where m is theta_1 and b is theta_0.
    return theta_0 + theta_1 * x

# print data to check if extraction was successful
x, y = extract_data(userPath)
print(f"X values: {x}\n")
print(f"Y values: {y}\n")

# print the coefficients
theta1, theta0 = calculate_coefficients(x, y)

# plot the data and the line of best fit
plot_data("Linear Regression Fit", x, y, theta1, theta0)

# apply batch gradient descent
theta1, theta0 = batch_gradient_descent(x, y, learning_rate, acceptable_error, delay)

print(f"\nTheta_1: {theta1}")
print(f"Theta_0: {theta0}\n")

# make prediction for a population of 35,000 people
print("The model predicts a profit of $", make_prediction(3.5, theta1, theta0) * 10000)

plot_data("Predicted Profit for 35,000 People", x, y, theta1, theta0)

# wait for a few seconds between predictions so my eyes do not bleed
sleep(5)

# make prediction for a population of 70,000 people
print("The model predicts a profit of $", make_prediction(7.0, theta1, theta0) * 10000)

plot_data("Predicted Profit for 70,000 People", x, y, theta1, theta0)






