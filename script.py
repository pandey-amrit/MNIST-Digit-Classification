import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC

def preprocess():
    """ 
    Input:
    Although this function doesn't have any input, you are required to load
    the MNIST data set from file 'mnist_all.mat'.

    Output:
    train_data: matrix of training set. Each row of train_data contains 
      feature vector of a image
    train_label: vector of label corresponding to each image in the training
      set
    validation_data: matrix of training set. Each row of validation_data 
      contains feature vector of a image
    validation_label: vector of label corresponding to each image in the 
      training set
    test_data: matrix of training set. Each row of test_data contains 
      feature vector of a image
    test_label: vector of label corresponding to each image in the testing
      set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    train_data_bias = np.insert(train_data, 0, 1, axis=1)  # Add bias term
    w = initialWeights.reshape((n_features + 1, 1))

    theta_n = sigmoid(np.dot(train_data_bias, w))
    error = (labeli * np.log(theta_n + 1e-10) + (1.0 - labeli) * np.log(1.0 - theta_n + 1e-10))  # Add a small constant for numerical stability
    error = -np.sum(error) / n_data

    error_grad = np.dot(train_data_bias.T, (theta_n - labeli))
    error_grad = error_grad.flatten() / n_data

    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
      
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
          
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    data = np.insert(data, 0, 1, axis=1)  # Add bias term
    label = sigmoid(np.dot(data, W))
    label = np.argmax(label, axis=1)
    label = label.reshape((data.shape[0], 1))

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    train_data_bias = np.insert(train_data, 0, 1, axis=1)  # Add bias term
    w = params.reshape((n_feature + 1, n_class))

    theta_nk = np.exp(np.dot(train_data_bias, w))
    sum_theta = np.sum(theta_nk, axis=1)
    sum_theta = sum_theta.reshape(sum_theta.shape[0], 1)
    theta_nk = theta_nk / sum_theta

    inner_error = labeli * np.log(theta_nk + 1e-10)  # Add a small constant for numerical stability
    error = -np.sum(inner_error) / n_data

    error_grad = np.dot(train_data_bias.T, (theta_nk - labeli)) / n_data
    error_grad = error_grad.flatten()

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    data = np.insert(data, 0, 1, axis=1)  # Add bias term
    
    label = np.exp(np.dot(data, W))
    sum_label = np.sum(label, axis=1)
    sum_label = sum_label.reshape(sum_label.shape[0], 1)
    label = label / sum_label
    
    label = np.argmax(label, axis=1)
    label = label.reshape((data.shape[0], 1))

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 300}  # Increased number of iterations
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    
    # Reshape initialWeights to a 1D array
    initialWeights_flattened = initialWeights.flatten() 
    
    nn_params = minimize(blrObjFunction, initialWeights_flattened, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################
# Using linear kernel (all other parameters are kept default)
clf = SVC(kernel='linear')
clf.fit(train_data, train_label.flatten())

train_acc = 100 * clf.score(train_data, train_label)
validation_acc = 100 * clf.score(validation_data, validation_label)
test_acc = 100 * clf.score(test_data, test_label)

print('\n Training set Accuracy (Linear Kernel):' + str(train_acc) + '%')
print('\n Validation set Accuracy (Linear Kernel):' + str(validation_acc) + '%')
print('\n Testing set Accuracy (Linear Kernel):' + str(test_acc) + '%')

# Using radial basis function with value of gamma setting to 1
clf = SVC(kernel='rbf', gamma=1.0)
clf.fit(train_data, train_label.flatten())

train_acc = 100 * clf.score(train_data, train_label)
validation_acc = 100 * clf.score(validation_data, validation_label)
test_acc = 100 * clf.score(test_data, test_label)

print('\n Training set Accuracy (RBF Kernel, gamma=1):' + str(train_acc) + '%')
print('\n Validation set Accuracy (RBF Kernel, gamma=1):' + str(validation_acc) + '%')
print('\n Testing set Accuracy (RBF Kernel, gamma=1):' + str(test_acc) + '%')

# Using radial basis function with value of gamma setting to default
clf = SVC(kernel='rbf')
clf.fit(train_data, train_label.flatten())

train_acc = 100 * clf.score(train_data, train_label)
validation_acc = 100 * clf.score(validation_data, validation_label)
test_acc = 100 * clf.score(test_data, test_label)

print('\n Training set Accuracy (RBF Kernel, default gamma):' + str(train_acc) + '%')
print('\n Validation set Accuracy (RBF Kernel, default gamma):' + str(validation_acc) + '%')
print('\n Testing set Accuracy (RBF Kernel, default gamma):' + str(test_acc) + '%')

# Using radial basis function with value of gamma setting to default and varying value of C
C_values = np.arange(1, 101, 10)  # C values from 1 to 100 with step 10
accuracy_values = []
for C in C_values:
    clf = SVC(kernel='rbf', C=C)
    clf.fit(train_data, train_label.flatten())
    test_acc = 100 * clf.score(test_data, test_label)
    accuracy_values.append(test_acc)

# Plot the graph of accuracy with respect to values of C (This needs to be done in your report)
# You can use matplotlib for plotting
import matplotlib.pyplot as plt
plt.plot(C_values, accuracy_values)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Accuracy vs C for RBF Kernel')
plt.show()

"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 300}  # Increased number of iterations

args_b = (train_data, Y)

# Flatten initialWeights_b before passing to minimize
initialWeights_b_flattened = initialWeights_b.flatten()

nn_params = minimize(mlrObjFunction, initialWeights_b_flattened, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy (Multi-class LR):' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')