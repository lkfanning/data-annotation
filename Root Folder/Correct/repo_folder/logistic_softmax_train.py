from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import get_data

def y2indicator(y, K):
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

Xtrain, Ytrain, Xtest, Ytest = get_data()
D = Xtrain.shape[1]
K = len(set(Ytrain) | set(Ytest))

# convert to indicator
Ytrain_ind = y2indicator(Ytrain, K)
Ytest_ind = y2indicator(Ytest, K)

# randomly initialize weights
W = np.random.randn(D, K)
b = np.zeros(K)

# make predictions
def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W, b):
    return softmax(X.dot(W) + b)

def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis=1)

# calculate the accuracy
def classification_rate(Y, P):
    return np.mean(Y == P)

def cross_entropy(Y, pY):
    return -np.sum(Y * np.log(pY)) / len(Y)


# train loop
train_costs = []
test_costs = []
initial_learning_rate = 0.001  # Starting learning rate
decay_rate = 0.9             # Rate at which the learning rate decreases
decay_steps = 1000           # Number of iterations after which to decay the learning rate

max_iterations = 10000
learning_rates = []          # To track learning rate changes over time

for i in range(max_iterations):
    # Calculate current learning rate with exponential decay
    learning_rate = initial_learning_rate * (decay_rate ** (i / decay_steps))
   
    # Apply minimum cap
    learning_rate = max(learning_rate, min_learning_rate)
    
    learning_rates.append(learning_rate)

    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)

    ctrain = cross_entropy(Ytrain_ind, pYtrain)
    ctest = cross_entropy(Ytest_ind, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    # gradient descent with the current learning rate
    W -= learning_rate*Xtrain.T.dot(pYtrain - Ytrain_ind)
    b -= learning_rate*(pYtrain - Ytrain_ind).sum(axis=0)
    
    if i % 1000 == 0:
        print(i, ctrain, ctest, "learning rate:", learning_rate)

print("Final train classification_rate:", classification_rate(Ytrain, predict(pYtrain)))
print("Final test classification_rate:", classification_rate(Ytest, predict(pYtest)))

# Plotting the cost functions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_costs, label='train cost')
plt.plot(test_costs, label='test cost')
plt.title('Cross Entropy Cost over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.legend()

# Plot learning rate over time
plt.subplot(1, 2, 2)
plt.plot(learning_rates, label='learning rate')
plt.title('Learning Rate Decay')
plt.xlabel('Iterations')
plt.ylabel('Learning Rate')
plt.legend()

plt.tight_layout()
plt.savefig('/tmp/outputs/training_results.png')
plt.show()
