import numpy as np
import matplotlib.pyplot as plt

# Function to compute Mean Squared Error
def compute_mse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

# Function to perform Gradient Descent and update weights
def gradient_descent(x, y, m_current, c_current, learning_rate):
    n = len(y)
    y_predicted = m_current * x + c_current
    # Calculate gradients
    gradient_m = (-2/n) * np.sum(x * (y - y_predicted))
    gradient_c = (-2/n) * np.sum(y - y_predicted)
    # Update weights
    m_new = m_current - learning_rate * gradient_m
    c_new = c_current - learning_rate * gradient_c
    return m_new, c_new

# Create a simulated dataset
np.random.seed(42)
x = np.random.uniform(50, 200, 100)
y = 5000 + 50 * x + np.random.normal(0, 500, 100)

# Initialize parameters
np.random.seed(42)
m = np.random.rand()
c = np.random.rand()
learning_rate = 0.0001
epochs = 10

# Training the model
errors = []
for epoch in range(epochs):
    y_pred = m * x + c
    mse = compute_mse(y, y_pred)
    errors.append(mse)
    print(f"Epoch {epoch+1}: MSE = {mse}, m = {m}, c = {c}")
    m, c = gradient_descent(x, y, m, c, learning_rate)

# Plotting the line of best fit
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, m * x + c, color='red', label='Best Fit Line')
plt.xlabel('Office Size (sq ft)')
plt.ylabel('Office Price')
plt.title('Office Size vs. Price')
plt.legend()
plt.show()

# Predicting the office price for size 100 sq. ft.
size = 100
predicted_price = m * size + c
print(f"Predicted office price for size {size} sq ft: {predicted_price}")
