import numpy as np
import matplotlib.pyplot as plt

duty_hours = [5, 6, 7, 8, 9]  # Example duty hours (in hours)
salaries = [2000, 2500, 3500, 4000, 5000]  # Corresponding salaries (in dollars)

# Convert the duty hours and salary data into numpy arrays
x = np.array(duty_hours)  # Duty hours as x values
y = np.array(salaries)  # Salaries as y values

# Calculate the mean (average) of the duty hours and salary
mean_x = np.mean(x)  # Mean of the duty hours
mean_y = np.mean(y)  # Mean of the salaries

# Calculate the slope (m) and intercept (b) for the regression line
numerator = np.sum((x - mean_x) * (y - mean_y))  # Top part of the slope formula
denominator = np.sum((x - mean_x) ** 2)          # Bottom part of the slope formula
m = numerator / denominator                      # The slope (m)
b = mean_y - m * mean_x                          # The intercept (b)

# Use the regression line to predict salary values based on duty hours
y_pred = m * x + b  # Predicted salary values using the regression line

# Calculate the Mean Squared Error (MSE) to evaluate the model's accuracy
mse = np.mean((y - y_pred) ** 2)  # Average squared difference between actual and predicted salaries

# Calculate the R-squared value to measure the goodness of fit
ss_total = np.sum((y - mean_y) ** 2)  # Total variation in salary
ss_residual = np.sum((y - y_pred) ** 2)  # Remaining variation after regression
r_squared = 1 - (ss_residual / ss_total)  # How well the model fits the data

# Output the results
print("\nLinear Regression Results")
print(f"Equation: y = {m:.2f}x + {b:.2f}")  # Display the regression equation
print(f"Mean Squared Error (MSE): {mse:.2f}")  # Display MSE
print(f"R-squared: {r_squared:.2f}")  # Display R-squared

# Plot the original data and the regression line
plt.scatter(x, y, color="green", label="Data Points")         # Plot original data points as blue dots
plt.plot(x, y_pred, color="red", label="Regression Line")    # Plot regression line in red
plt.xlabel("Duty Hours (hours)")  # Label for the x-axis
plt.ylabel("Salary (dollars)")  # Label for the y-axis
plt.title("Linear Regression: Duty Hours vs Salary")  # Title of the plot
plt.legend()  # Show legend
plt.grid(True)  # Show grid for better readability
plt.show()  # Display the plot