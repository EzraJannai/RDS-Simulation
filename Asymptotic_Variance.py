import numpy as np
import matplotlib.pyplot as plt

# Define the first function
def my_function(z, h_1):
    #S is the expected value of the size biased dist. when set to 0 we look at the case were there is a guaranteed alternate referral
    S = 0
    q1 = ((h_1*z)/(1 + h_1*z))**(S-1)
    q2 = ((1 + (h_1 - 1)*z)/(1 + h_1*z))**(S-1)
    
    if S != 0:
      f2 = (z*(1 + z)*(1 + (-1 + 2*h_1)*z))/((S-1))
      f3 = ((-2 + q1 + q2)*z*(-1 + q2 + (-1 + h_1)*(-1 + q2)*z + h_1*(-1 + q1)*(z**2)))/((S-1)*(-1 + q1)*(-1 + q2))
    else:
      f3 = (2 + 2 * z * (-1 + h_1 + h_1 * z))
      f2 = ((1 + z) * (1 + (-1 + 2 * h_1) * z))
    return f3/f2

# Define the second function
def my_function_2(z, h_1):
    q1 = ((h_1*z)/(1 + h_1*z))**(5-1)
    q2 = ((1 + (h_1 - 1)*z)/(1 + h_1*z))**(5-1)
    f2 = (((h_1*(1 + z)*(1 + h_1*z*(1 + (-1 + h_1)*z)))) / ((5-1) * (z ** 2)))
    f3 = (h_1*(-2 + q1 + q2)*(-1 + q1 + h_1*(-1 + q2)*(1 + (-1 + h_1)*z)))/((5-1) * (-1 + q1)*(-1 + q2)*z)
    return f3/f2

# Define the function that solves the first function for z = 1
def solve_for_z_1(h_1):
    return np.ones_like(h_1)

# Define the function that solves the second function for z = 1
def solve_for_z_2(h_1):
    positive_solution = (-h_1 + np.sqrt(h_1) * np.sqrt(-4 + 5 * h_1)) / (2 * (-h_1 + h_1**2))
    negative_solution = (-h_1 - np.sqrt(h_1) * np.sqrt(-4 + 5 * h_1)) / (2 * (-h_1 + h_1**2))

    positive_solution = np.clip(positive_solution, -20, 20)

    # Create masks for the ranges
    mask_negative = np.logical_and(h_1 >= 0.75, h_1 <= 0.98)
    mask_positive = h_1 >= 0.75

    # Apply the masks
    negative_solution = np.where(mask_negative, np.clip(negative_solution, -20, 20), np.nan)
    positive_solution = np.where(mask_positive, positive_solution, np.nan)

    return positive_solution, negative_solution

# Define the range and number of points for z and h_1
z_min = 0.1
z_max = 10.0
h_1_min = 0.1
h_1_max = 10.0
num_points = 500

# Generate a grid of z and h_1 values
z_values = np.logspace(np.log10(z_min), np.log10(z_max), num_points)
h_1_values = np.logspace(np.log10(h_1_min), np.log10(h_1_max), num_points)
z_grid, h_1_grid = np.meshgrid(z_values, h_1_values)

# Evaluate the first function for each combination of z and h_1
result = my_function(z_grid, h_1_grid)

# Evaluate the second function for each combination of z and h_1
result_2 = my_function_2(z_grid, h_1_grid)

# Create a condition for the region to be white
condition = (h_1_grid - 1)*z_grid + 1 < 0
result = np.where(condition, np.nan, result)
result_2 = np.where(condition, np.nan, result_2)

# Create the heatmap plot for the first function
plt.figure(figsize=(8, 6))
plt.pcolormesh(h_1_grid, z_grid, result, shading='auto', cmap='coolwarm', vmin=0, vmax=2)
plt.colorbar(label='')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$h_1$')
plt.ylabel('$z$')
plt.title('$\\frac{{var(\\hat{z})_3}}{{var(\\hat{z})_2}}$')
plt.grid(True)

# Plot the line where the function is equal to 1
z_values_1 = solve_for_z_1(h_1_values)

# Set the z-axis range
plt.ylim([0.1, 10])
plt.savefig('plot1.png')
# Show the first plot
plt.show()

# Create the heatmap plot for the second function
plt.figure(figsize=(8, 6))
plt.pcolormesh(h_1_grid, z_grid, result_2, shading='auto', cmap='coolwarm', vmin=0, vmax=2)
plt.colorbar(label='')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$h_1$')
plt.ylabel('$z$')
plt.title('$\\frac{{var(\\hat{h_1})_3}}{{var(\\hat{h_1})_2}}$')
plt.grid(True)

# Plot the lines where the function is equal to 1
# Set the z-axis range
plt.ylim([0.1, 10])
plt.savefig('plot1.png')
# Show the second plot
plt.show()
