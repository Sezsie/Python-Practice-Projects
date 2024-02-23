# Linear Regression Calculations and Visualization
# AUTHOR: Garrett Thrower
# Last Modified: 2/3/2024

# This script calculates the coefficients for a multivariate linear regression model using the least squares method.
# All done manually, since the point is to understand the math behind it.
# It then plots the data and the line of best fit using matplotlib.

from time import sleep
import matplotlib.pyplot as plt
import re

# parameters for the gradient descent, tweak them as you see fit

acceptable_error = 0.01 # the error threshold that will stop the gradient descent
learning_rate = 0.01 # how much the coefficients will be updated in each iteration
delay = 0.01 # delay between iterations in seconds

# this is the path to the file with the data on my computer, but probably not on yours. Change it to the path of your file
userPath = "C:/Users/User1/Desktop/boston.txt" 


def preprocess_data(file_path):
    """
    Reads a text file, removes every odd-line newline character, 
    and replaces multi-character whitespace with a single space.

    Parameters:
    - file_path: The path to the text file to be processed.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Process each line
    new_lines = []
    for i, line in enumerate(lines):
        if i % 2 == 0:  # Check if the line number is odd (considering 0-indexing)
            # Remove newline character and trailing spaces, then add to new_lines without a newline
            new_lines.append(line.strip())
        else:
            # For even lines, remove newline characters, condense spaces, and add a space at the end for merging
            new_lines[-1] += re.sub(r'\s+', ' ', line).strip() + " "
    
    # Write the processed lines back to the file
    with open(file_path, 'w') as file:
        for line in new_lines:
            file.write(line + "\n")  # Add back a newline character at the end of each merged line


preprocess_data(userPath)






