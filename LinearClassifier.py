import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data Preparation
inputMat = pd.read_csv('m_creditcard_24650.csv')  # returns dataframe
npMat = np.array(inputMat[['V2', 'V11', 'Class']])
inputInstances = npMat[:, 0:2]
# for the bias
inputInstances = np.c_[np.ones(len(inputInstances)), inputInstances]  # For the bias
output = npMat[:, 2]
# begin with random values for the weights
weightVector = np.random.rand(3)
print('Initial weight values: ', weightVector)
learning_rate = 0.2
Iterations = 0
errors = 0
converged = False
while not converged:
    missclassifications = 0  # calculate errors over an iteration
    for r in range(0, inputInstances.shape[0]):
        instance = inputInstances[r]
        dotProd = np.dot(instance, weightVector)
        activation = 1 if (dotProd > 0)else 0
        errors += 1 if (activation != output[r]) else 0
        error = activation - output[r]
        weightVector -= learning_rate * error * instance
        missclassifications += abs(error)
    Iterations += 1
    if missclassifications <= 12 or Iterations >= 5000:
        print('Converged after :', Iterations, ' iterations.')
        print(missclassifications)
        converged = True

print('Weight Vector: ', weightVector)

# plot the data; blue for genuine and red for fraud
for r in range(0, inputInstances.shape[0]):
    if output[r] == 1:
        plt.scatter(inputInstances[r][1], inputInstances[r][2], marker='x', color='r')
    else:
        plt.scatter(inputInstances[r][1], inputInstances[r][2], marker='o', color='b')
# plot the line
x = inputInstances[:, 2]
y = -((weightVector[1] / weightVector[2]) * x) - (weightVector[0] / weightVector[2])
plt.plot(x, y)
plt.show()
