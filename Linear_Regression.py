import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from  sklearn.model_selection import train_test_split

# data preparation
inputMat = pd.read_csv('DL.csv', low_memory=False)
npMat = np.array(inputMat[['DISTANCE', 'AIR_TIME']].dropna())

x = npMat[:, 0]
y = npMat[:, 1]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=0.2, random_state=101)
# for the bias
X_train = np.c_[np.ones(len(X_train)), X_train]
X_test = np.c_[np.ones(len(X_test)), X_test]
# data normalization
X_train[:, 1] = (X_train[:, 1] - np.min(X_train[:, 1])) / (
    np.max(X_train[:, 1]) - np.min(X_train[:, 1]))

X_test[:, 1] = (X_test[:, 1] - np.min(X_test[:, 1])) / (
    np.max(X_test[:, 1]) - np.min(X_test[:, 1]))


def train_sample():
    iterations = 0
    rate = 0.2
    weightVector = np.random.rand(2)  # to produce 2 random values
    weightVectorOld = [100, 100]
    while np.abs(weightVector[0] - weightVectorOld[0]) > 0.001 or np.abs(weightVector[1] - weightVectorOld[1]) > 0.001:
        weightVectorOld = weightVector
        for r in range(0, X_train.shape[0]):
            instance = X_train[r]
            dotProd = np.dot(weightVector, instance) - Y_train[r]
            # Update the weights
            weightVector = weightVector - rate * instance * dotProd
        iterations += 1;

    print('Number of iterations to converge: ', iterations)
    # plot the data and the line
    x = np.linspace(np.min(X_train[:, 1]), np.max(X_train[:, 1]))
    y = weightVector[0] + weightVector[1] * x
    plt.plot(X_train[:, 1], Y_train, 'bo', x, y, 'g-')
    plt.show()
    return weightVector


def test():
    w = train_sample()
    return mean_square_error(X_test[:, 1] * w[1] + w[0], Y_test)


def mean_square_error(y_predicted, y_actual):
    return np.square(y_predicted - y_actual).mean()


print('Mean Square Error over test data:', test())
