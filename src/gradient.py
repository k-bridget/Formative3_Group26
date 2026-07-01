import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime


#===========================
# Declaration of x and y
#===========================

X = np.array([
    [1,3],
    [4,10]
], dtype=float)

y = np.array([
    [5],
    [6]
], dtype=float)


#========================
# Initialize parameters (weight & bias)
#========================

m = np.array([
    [-1],
    [2]
], dtype=float)

b = np.array([
    [1],
    [1]
], dtype=float)


#=============
# Prediction
#=============

def predict(X, m, b):
    return X @ m + b

predict(X,m,b)


#======================
# Cost function (MSE) | Calculate how wrong is the model
#======================

def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)


#============
# Gradient (How we should change our parameters)
#============

def gradients(X, y, y_hat):

    n = len(y)

    error = y_hat - y

    grad_m = (2/n) * X.T @ error

    grad_b = (2/n) * np.sum(error)

    return grad_m, grad_b

# This decide whether we increase or decrease the parameter value


#====================
# Scipy derivative
#====================

def equation(x):
    return x[0]**2 + 3*x[0]

point = np.array([2.0])

gradient = approx_fprime(
    point,
    equation,
    epsilon=1e-6
)

print(gradient)


#================
# History list
#================

m_history = []

b_history = []

cost_history = []


#=========================
# Gradient descent loop
#=========================

learning_rate = 0.01

iterations = 20

for i in range(iterations):

    y_hat = predict(X,m,b)

    cost = mse(y,y_hat)

    grad_m, grad_b = gradients(
        X,
        y,
        y_hat
    )

    m = m - learning_rate*grad_m

    b = b - learning_rate*grad_b

    m_history.append(m.copy())

    b_history.append(b.copy())

    cost_history.append(cost)

    print(f"Iteration {i+1}")

    print("Prediction")

    print(y_hat)

    print("Cost")

    print(cost)

    print("Weights")

    print(m)

    print("Bias")

    print(b)

    print("--------------------")


#===================
# plot m (weight)
#===================

m1 = [x[0][0] for x in m_history]

m2 = [x[1][0] for x in m_history]

plt.plot(m1,label="m1")

plt.plot(m2,label="m2")

plt.xlabel("Iteration")

plt.ylabel("Weight")

plt.title("Weights During Gradient Descent")

plt.legend()

plt.show()


#=================
# Plot b (bias)
#=================

b1 = [x[0][0] for x in b_history]

b2 = [x[1][0] for x in b_history]

plt.plot(b1,label="b1")

plt.plot(b2,label="b2")

plt.xlabel("Iteration")

plt.ylabel("Bias")

plt.title("Bias During Gradient Descent")

plt.legend()

plt.show()


#================
# plot error
#================

plt.plot(cost_history)

plt.xlabel("Iteration")

plt.ylabel("MSE")

plt.title("Cost Function")

plt.show()