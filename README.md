# linear_regression_1
Simple linear regression and gradient descent implementation in Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Load dataset
df = pd.read_csv("heart.csv")
train_set = np.array(df)

# 2) Separate features (X) and target (Y)
x_train = train_set[:, 0]    # first column as input
y_train = train_set[:, 7]    # eighth column as output

# 3) Linear model: y_hat = w1 * x + w0
w0 = 0.0
w1_candidates = np.linspace(-100, 100, 100)

def linear_regression(x, w0, w1):
    return w1 * x + w0

# 4) Loss function (Mean Squared Error)
def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)

# 5) Compute MSE for each candidate w1
mse_loss = []
for w1 in w1_candidates:
    y_hat = linear_regression(x_train, w0, w1)
    mse_loss.append(mse(y_train, y_hat))

mse_loss = np.array(mse_loss)
min_index = np.argmin(mse_loss)
best_w1 = w1_candidates[min_index]
best_loss = mse_loss[min_index]

print(f"Best w1: {best_w1:.4f}, Min MSE: {best_loss:.6f}")

# Plot MSE vs w1
plt.figure(figsize=(6,4))
plt.plot(w1_candidates, mse_loss, label="MSE vs w1")
plt.scatter(best_w1, best_loss, color="red", label="Minimum")
plt.xlabel("w1")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.show()

# 6) Example: Gradient descent on f(x) = x^2
def funcx2(x):
    return x**2

def gradfuncx2(x):
    return 2*x

x = np.linspace(-5, 5, 200)
plt.figure(figsize=(6,4))
plt.plot(x, funcx2(x), label="y = x^2")
plt.plot(x, gradfuncx2(x), label="grad = 2x")
plt.legend()
plt.tight_layout()
plt.show()

# Gradient descent in 1D
def gradient_descent_1d(gradfunc, xi, eta, n):
    for _ in range(n):
        grad = gradfunc(xi)
        xi -= eta * grad   # update in the opposite direction of gradient
    return xi

final_x = gradient_descent_1d(gradfuncx2, xi=4.0, eta=0.2, n=50)
print(f"Final x (1D GD): {final_x:.6f}")

# 7) Example: Gradient descent in 2D on f(x,y) = x^2 + y^2
def func2d(x, y):
    return x**2 + y**2

def gradfunc2d(x, y):
    return np.array([2*x, 2*y])

# Gradient descent in 2D
def gradient_descent_2d(gradfunc, xi, yi, eta, n):
    for _ in range(n):
        grad = gradfunc(xi, yi)
        xi -= eta * grad[0]
        yi -= eta * grad[1]
    return xi, yi

final_x2d, final_y2d = gradient_descent_2d(gradfunc2d, xi=-4.0, yi=3.0, eta=0.1, n=100)
print(f"Final (x,y) (2D GD): ({final_x2d:.6f}, {final_y2d:.6f})")
