import numpy as np
import matplotlib.pyplot as plt

# To iterate over JPG files in a directory
import os

# To read and manipulate images
import cv2

# CVXOPT library version
# 'matrix' is the library representation of matrices
# 'solvers' contains the actual methods for optimisation
from cvxopt import matrix, solvers

# 'svm' is the class to represent SVM models in scikit-learn
# 'confusion_matrix' calculates the confusion matrix for a given set of actual values and predicted values
from sklearn import svm
from sklearn.metrics import confusion_matrix

# FUNCTIONS TO CONSTRUCT TRAINING DATA FROM JPG FILES
# Iterates over all the files in a folder and forms the feature vector for each image
def get_img_data(dir_path):
    
    # Each image is 16 x 16 x 3. Therefore, after flattening, each feature vector's dimension is 768
    feature_len = 768
    desired_width = 16
    desired_height = 16

    # Contains the feature vectors. Each column represents the feature vector of a particular example
    data = np.empty((feature_len, 0))

    # 'files' contains the names of the files in the directory
    files = os.listdir(dir_path)

    for file in files:
        
        # image_path is the relative path of the image (relative to this Jupyter Notebook)
        image_path = os.path.join(dir_path, file)
        image = cv2.imread(image_path)

        # Resizing the image to 16 x 16 x 3
        image = cv2.resize(image, (desired_width, desired_height))
        
        # Converts the OpenCV Image object to NumPy array
        np_array = np.array(image)

        # Flatten the 3D array to a 1D vector
        np_vector = np_array.flatten().reshape((feature_len, 1))
        
        # Add this feature vector to the data array
        data = np.concatenate((data, np_vector), axis=1)
    
    # Normalise the RGB values for each example
    data = data/255
    
    return data

# Constructs the feature vector array and label array given the path to the folders containing images of each class
def format_data(dir_path_1, dir_path_2):

    class_1_data = get_img_data(dir_path_1)         # These examples are given class value +1
    class_2_data = get_img_data(dir_path_2)         # These examples are given class value -1

    pos_labels = np.full((1, class_1_data.shape[1]), 1.0)
    neg_labels = np.full((1, class_2_data.shape[1]), -1.0)
    
    data_x = np.concatenate((class_1_data, class_2_data), axis=1)
    data_y = np.concatenate((pos_labels, neg_labels), axis=1)

    return data_x, data_y

def format_data_multi(dir_path_list):

    label = 0.0
    data_y = np.empty((1,0))
    data_x = np.empty((768,0))
    for path in dir_path_list:
        imgs = get_img_data(path)
        labels = np.full((1, imgs.shape[1]), label)

        data_x = np.concatenate((data_x, imgs), axis=1)
        data_y = np.concatenate((data_y, labels), axis=1)

        label = label + 1.0


    # This shuffles the training examples.
    # This is necessary for k-fold cross validation question since we are using only a subset of training data there
    combined = np.concatenate((data_x, data_y), axis=0)
    np.random.shuffle(combined.T)       # np.random.shuffle() shuffles along the first dimension. Hence, transpose is taken first
    
    # Extracts the data_x and data_y from the combined array
    data_x, data_y = combined[0:768], combined[768:769]
    
    return data_x, data_y

# CLASS FOR SVM MODELS OPTIMISED USING CVXOPT PACKAGE
class SVM_Cvxopt:
    
    def __init__(self, training_data_x, training_data_y, kernel, C, gamma=0):
        
        '''
        training_data_x, training_data_y : ndarrays | images and labels respectively
        C : float | hyperparameter to determine importance of slack variables
        kernel : string | 'linear' or 'gaussian'
        gamma : float | used to compute gaussian kernel | needed only when kernel is gaussian
        '''

        self.training_data_x = training_data_x          # (768 x m)
        self.training_data_y = training_data_y          # (1 x m)
        self.kernel = kernel                            # 'linear' or 'gaussian'
        self.C = C                                      # scalar
        self.gamma = gamma                              # scalar
        self.m = self.training_data_x.shape[1]          # scalar | number of training examples
        self.p = np.zeros((self.m, self.m))             # (m x m)
        self.q = np.zeros((self.m, 1))                  # (m, 1)
        self.g = np.zeros((2*self.m, self.m))           # (2m x m)
        self.a = np.zeros((1, self.m))                  # (1 x m)
        self.b = np.zeros((1, 1))                       # scalar
        self.h = np.zeros((2*self.m,))                  # (2m x 1)
        self.alpha = np.zeros((self.m, 1))              # (1 x m)
        self.n_sv=0                                     # scalar
        self.sv_idx = []

        self.get_matrices()                             # computes p, q, g, h, a, b
        self.get_optimal_alpha()                        # optimises the dual to get alpha params

    # computes p, q, g, h, a, b
    def get_matrices(self):
        
        self.q = np.full((self.m,1), -1.0)

        self.b = np.array([[0.0]])

        matrix_g_pos = np.eye(self.m, dtype=float)
        matrix_g_neg = (-1) * np.eye(self.m, dtype=float)
        self.g = np.concatenate((matrix_g_pos, matrix_g_neg), axis=0)
        
        matrix_h_C = np.full((self.m, 1), self.C)
        matrix_h_0 = np.full((self.m, 1), 0.0)
        self.h = np.concatenate((matrix_h_C, matrix_h_0), axis=0)

        self.a = self.training_data_y

        # computation for p
        kernel_matrix = self.get_kernel_matrix(self.training_data_x, self.training_data_x)
        y_product = np.matmul(self.training_data_y.T, self.training_data_y)
        self.p = kernel_matrix * y_product      
            
    # optmises the dual to get alpha parameters
    def get_optimal_alpha(self):
        
        # in-built solver provided by CVXOPT
        solution = solvers.qp(matrix(self.p), matrix(self.q), matrix(self.g), matrix(self.h), matrix(self.a), matrix(self.b))
        
        # optimal alpha parameters
        self.alpha = np.array(solution['x'].T)
        
        # Approximates those alpha parameters to 0 or 1 which are very near to 0 or 1 respectively
        # number of alpha parameters s.t 0 < alpha_param <= C are considered as support vectors
        for i in range(self.m):
            if (self.alpha[0, i] > 5 * 1e-9) and (self.alpha[0, i] < (1 - 1e-10)):
                self.n_sv += 1
                self.sv_idx.append(i)
        
    # computes the w vector | only for linear kernel
    # w = (768 x 1)
    def get_w(self):
        if self.kernel == 'linear':
            w = np.sum((self.alpha*self.training_data_y)*self.training_data_x, axis=1).reshape((768, 1))
        return w

    # computes the kernel matrix for given matrices x and y
    # x = (768 x m) and y = (768 x n)
    # kernel_matrix = (m x n)
    def get_kernel_matrix(self, x, y):
        x_T, y_T = x.T, y.T
        
        # computes the matrix if the kernel is gaussian
        if self.kernel == 'gaussian':
            norm_squared_x = np.sum(x_T ** 2, axis=1, keepdims=True)
            norm_squared_y = np.sum(y_T ** 2, axis=1)
            pairwise_distances = -2 * np.dot(x_T, y_T.T) + norm_squared_x + norm_squared_y
            kernel_matrix = np.exp(-self.gamma * pairwise_distances)
        
        # computes the matrix if the kernel is linear
        elif self.kernel == 'linear':
            kernel_matrix = np.matmul(x_T, y)
        
        return kernel_matrix
    
    # computes 'b'
    # since none of the obtained alpha are exactly 0, for computation purpose, all examples are considered to be support vectors
    def get_b(self):

        kernel_matrix = self.get_kernel_matrix(self.training_data_x, self.training_data_x)      # computes the kernel matrix
        kernel_vector = np.sum(kernel_matrix, axis=1, keepdims=True)                            # sums over rows
        
        temp = self.alpha * self.training_data_y

        sigma_y = np.sum(self.training_data_y, axis=1, keepdims=True)[0, 0]                     # computes the sum of all labels

        b = (sigma_y - np.matmul(temp, kernel_vector)[0, 0]) / (self.m)
    
        return b
    
    # returns a list of predictions for examples in data_x
    def get_predictions(self, data_x, data_y):

        temp_1 = self.alpha * self.training_data_y
        temp_2 = self.get_kernel_matrix(self.training_data_x, data_x)
        
        b = self.get_b()

        # for example x, prediction is the sign of {dot(w, x) + b}
        predictions = np.sign(np.matmul(temp_1, temp_2) + b)

        return predictions

    # returns the percentage accuracy over data_x
    # similar to get_predictions() function. Just returns the percentage of accurate predictions, instead of list of predictions
    def get_accuracy(self, data_x, data_y):
        
        examples = data_x.shape[1]

        temp_1 = self.alpha * self.training_data_y
        temp_2 = self.get_kernel_matrix(self.training_data_x, data_x)
        
        b = self.get_b()

        predictions = np.sign(np.matmul(temp_1, temp_2) + b)

        accuracy = np.sum(predictions == data_y) / examples

        return accuracy
    
# CLASS FOR SVM MODELS OPTIMISED USING SCIKIT LEARN
class SVM_Sklearn:

    def __init__(self, training_data_x, training_data_y, kernel, C, gamma=0):
        
        '''
        training_data_x, training_data_y : ndarrays | images and labels respectively
        C : float | hyperparameter to determine importance of slack variables
        kernel : string | 'linear' or 'rbf' (gaussian)
        gamma : float | used to compute gaussian kernel | needed only when kernel is gaussian
        '''

        self.training_data_x = training_data_x      # (768 x m)
        self.training_data_y = training_data_y      # (1 x m)
        self.kernel = kernel                        # 'linear' or 'rbf'
        self.C = C                                  # scalar
        self.gamma = gamma                          # scalar
        self.m = self.training_data_x.shape[1]      # scalar | number of training examples
        
        self.model_svm()                            # optimises th dual of the svm
    
    # models the SVM problem and optmises the dual problem
    # all the computed quantities are properties of 'svm_svc' object
    def model_svm(self):
        
        self.svm_svc = svm.SVC(C=self.C, kernel=self.kernel, gamma=self.gamma)                  # constructs the model
        self.svm_svc.fit(self.training_data_x.T, self.training_data_y.reshape(self.m,))         # optimises the dual

    # returns the list of support vectors
    def get_support_vectors(self):
        return self.svm_svc.support_vectors_
    
    # returns a list of predictions for examples in data_x
    def get_predictions(self, data_x):
        temp = data_x.T
        predictions = self.svm_svc.predict(temp)
        predictions = np.array(predictions, ndmin=2)
        return predictions
    
    # returns the percentage accuracy over data_x
    # similar to get_predictions() function. Just returns the percentage of accurate predictions, instead of list of predictions
    def get_accuracy(self, data_x, data_y):
        examples = data_x.shape[1]
        temp = data_x.T
        predictions = self.svm_svc.predict(temp)
        predictions = np.array(predictions, ndmin=2)
        accuracy = np.sum(predictions == data_y)/examples
        return accuracy

'''
TRAINING AND VALIDATION DATA FOR BINARY CLASSIFICATION
Entry No. = 2021EE10627
Therefore, given classes are 1 and 2
'''
# training data for binary classification
training_data_x, training_data_y = format_data('train/1/', 'train/2/')

# validation data for binary classification
validation_data_x, validation_data_y = format_data('val/1/', 'val/2/')

'''
BINARY CLASSIFICATION USING CVXOPT
LINEAR KERNEL
'''
svm_cvxopt_linear = SVM_Cvxopt(training_data_x, training_data_y, 'linear', 1.0)

w = svm_cvxopt_linear.get_w()       # computes w vector
b = svm_cvxopt_linear.get_b()       # computes optimal bias (scalar 'b')

accuracy_linear_cvx_train = svm_cvxopt_linear.get_accuracy(training_data_x, training_data_y)            # accuracy over training data
accuracy_linear_cvx_valid = svm_cvxopt_linear.get_accuracy(validation_data_x, validation_data_y)        # accuracy over validation data

# shows the image using OpenCV given a NumPy row matrix
def save_image(name, x):

    # since 16 x 16 is a very small image, I scale each pixel to occupy 'scaling_factor' times area for better visualisation
    scaling_factor = 20
    
    # since we had originally normalised the image, we denormalise it here
    x *= 255

    # OpenCV takes RGB values to be unsigned 8 bit integers. I round off each value to nearest integer and then convert it into uint8 data type
    x_uint8 = np.round(x).astype(np.uint8)
    
    # reshapes the vector into 16x16x3 matrix
    img_mat = x_uint8.reshape((16, 16, 3))
    
    # converts the matrix into an OpenCV image
    img = cv2.cvtColor(img_mat, cv2.COLOR_RGB2BGR)    
    # enlarges the image to occupy more area while preserving the resolution
    en_img = cv2.resize(img, None, fx = scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_NEAREST)
    
    # image is finally drawn on a canvas
    # I define that canvas here
    canvas_size = (en_img.shape[0], en_img.shape[1])
    canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
    
    # place the image on the canvas
    x_offset, y_offset = 0, 0
    canvas[y_offset:y_offset + en_img.shape[0], x_offset:x_offset + en_img.shape[1]] = en_img

    # show tha image and indefinitely wait for a key press before closing the window
    cv2.imwrite(f'{name}.png', canvas)

# TOP-5 IMAGES 'W' - LINEAR KERNEL
# list containing indices of 6 examples with largest alpha paraemeters
top_alpha_indices_linear = np.argsort(svm_cvxopt_linear.alpha.flatten())[-6:]

# shows the 6 images with largest alpha parameters
counter = 1
for idx in top_alpha_indices_linear:
    img_vector = training_data_x.T[idx]
    save_image(f'lin-{counter}', img_vector)
    counter += 1

# values in 'w' can exceed 1. Therefore, while denormalising the vector, the values can exceed 255 (out of range)
# therefore, I linearly map the values of weight vector to be between 0 and 1
w_min, w_max = np.min(w), np.max(w)
w_scaled = (w - w_min) / (w_min-w_max)
save_image('w', w_scaled)                  # shows image formed by 'w' in RGB format

'''
BINARY CLASSIFICATION USING CVXOPT
GAUSSIAN KERNEL
'''
svm_cvxopt_gaussian = SVM_Cvxopt(training_data_x, training_data_y, 'gaussian', 1.0, 0.001)

accuracy_gaussian_cvx_train = svm_cvxopt_gaussian.get_accuracy(training_data_x, training_data_y)        # accuracy over training data
accuracy_gaussian_cvx_valid = svm_cvxopt_gaussian.get_accuracy(validation_data_x, validation_data_y)    # accuracy over validation data

# TOP 6 IMAGES - GAUSSIAN KERNEL
# list containing indices of 6 examples with largest alpha paraemeters
top_alpha_indices_gaussian = np.argsort(svm_cvxopt_gaussian.alpha.flatten())[-6:]

# shows the 6 images with largest alpha parameters
counter = 1
for idx in top_alpha_indices_gaussian:
    img_vector = training_data_x.T[idx]
    save_image(f'gau-{counter}', img_vector)
    counter += 1

'''
BINARY CLASSIFICATION USING SCIKIT LEARN
LINEAR KERNEL
'''

svm_sklearn_linear = SVM_Sklearn(training_data_x, training_data_y, 'linear', 1)

sv_linear_sklearn = svm_sklearn_linear.get_support_vectors()            # list of support vectors

accuracy_linear_sklearn_train = svm_sklearn_linear.get_accuracy(training_data_x, training_data_y)           # accuracy over training set
accuracy_linear_sklearn_valid = svm_sklearn_linear.get_accuracy(validation_data_x, validation_data_y)       # accuracy over validation set

# gets the 'w' vector and 'b' learnt by SVM using sklearn
w_sklearn = svm_sklearn_linear.svm_svc.coef_
b_sklearn = svm_sklearn_linear.svm_svc.intercept_

'''
BINARY CLASSIFICATION USING SCIKIT LEARN
GAUSSIAN (RBF) KERNEL
'''

svm_sklearn_gaussian = SVM_Sklearn(training_data_x, training_data_y, 'rbf', 1, 0.001)

sv_gaussian_sklearn = svm_sklearn_gaussian.get_support_vectors()        # list of support vectors

accuracy_gaussian_sklearn_train = svm_sklearn_gaussian.get_accuracy(training_data_x, training_data_y)           # accuracy over training data
accuracy_gaussian_sklearn_valid = svm_sklearn_gaussian.get_accuracy(validation_data_x, validation_data_y)       # accuracy over validation data

# COMMON SUPPORT VECTORS
sv_lin_cvx = np.sort(svm_cvxopt_linear.sv_idx)       # Sorted list of indices of support vectors of linear SVM learnt using CVXOPT
sv_gau_cvx = np.sort(svm_cvxopt_gaussian.sv_idx)     # Sorted list of indices of support vectors of gaussian SVM learnt using CVXOPT

sv_lin_skl = np.sort(svm_sklearn_linear.svm_svc.support_)       # Sorted list of indices of support vectors of linear SVM learnt using sklearn
sv_gau_skl = np.sort(svm_sklearn_gaussian.svm_svc.support_)     # Sorted list of indices of support vectors of gaussian SVM learnt using sklearn

cvx_lin_gau = len(np.intersect1d(sv_lin_cvx, sv_gau_cvx))       # number of SVs common to linear and gaussian SVM learnt using CVXOPT
skl_lin_gau = len(np.intersect1d(sv_lin_skl, sv_gau_skl))       # number of SVs common to linear and gaussian SVM learnt using sklearn

lin_cvx_skl = len(np.intersect1d(sv_lin_cvx, sv_lin_skl))       # number of SVs common to linear SVMs learnt using CVXOPT and sklearn
gau_cvx_skl = len(np.intersect1d(sv_gau_cvx, sv_gau_skl))       # number of SVs common to gaussian SVMs learnt using CVXOPT and sklearn

# TRAINING AND VALIDATION DATA FOR MULTICLASS CLASSIFICATION
# training data for multi-class classification
training_paths = ['train/0/', 'train/1/', 'train/2/', 'train/3/', 'train/4/', 'train/5/']
training_data_x_multi, training_data_y_multi = format_data_multi(training_paths)

# validation data for multi-class classification
validation_paths = ['val/0/', 'val/1/', 'val/2/', 'val/3/', 'val/4/', 'val/5/']
validation_data_x_multi, validation_data_y_multi = format_data_multi(validation_paths)

'''
MULTI CLASSIFICATION USING CVXOPT
GAUSSIAN KERNEL
WE HAVE COMB(6, 2) = 15 CLASSIFIERS
'''

# TRAINING DATA FOR EACH CLASSIFIER
training_data_x_dict = {}       # contains training data_x mapped to each of the 15 (i, j) class pairs
training_data_y_dict = {}       # contains traiing data_y mapped to each of the 15 (i, j) class pairs

for i in range(0, 6):
    for j in range(i+1, 6):
        
        # filters the data for examples of class i or j
        data_x_training = training_data_x_multi.T[(training_data_y_multi[0] == i) | (training_data_y_multi[0] == j)].T
        
        data_y_training = training_data_y_multi[0][(training_data_y_multi[0] == i) | (training_data_y_multi[0] == j)]
        data_y_training = data_y_training.reshape((1, data_y_training.shape[0]))
        # treats class i as 1 and class j as -1
        data_y_training = np.where(data_y_training == i, 1.0, -1.0)

        training_data_x_dict[(i, j)] = data_x_training
        training_data_y_dict[(i, j)] = data_y_training

# VALIDATION DATA FOR EACH CLASSIFIER
validation_data_x_dict = {}     # contains validation data_x mapped to each of the 15 (i, j) class pairs
validation_data_y_dict = {}     # contains validation data_y mapped to each of the 15 (i, j) class pairs

for i in range(0, 6):
    for j in range(i+1, 6):
        
        # filters the data for examples of class i or j
        data_x_validation = validation_data_x_multi.T[(validation_data_y_multi[0] == i) | (validation_data_y_multi[0] == j)].T
        
        data_y_validation = validation_data_y_multi[0][(validation_data_y_multi[0] == i) | (validation_data_y_multi[0] == j)]
        data_y_validation = data_y_validation.reshape((1, data_y_validation.shape[0]))
        # treats class i as 1 and class j as -1
        data_y_validation = np.where(data_y_validation == i, 1.0, -1.0)

        validation_data_x_dict[(i, j)] = data_x_validation
        validation_data_y_dict[(i, j)] = data_y_validation

# LEARNING THE CLASSIFIERS
# contains the SVM model objects corresponding to each of the 15 (i, j) class pairs
models_dict = {}

for i in range(0, 6):
    for j in range(i+1, 6):
        
        data_x, data_y = training_data_x_dict[(i, j)], training_data_y_dict[(i, j)]
        
        model = SVM_Cvxopt(data_x, data_y, 'gaussian', 1.0, 0.001)
        
        models_dict[(i, j)] = model

# ACCURACY ON TRAINING DATA
# 2D list containing the predictions over the entire training set made by each classifier
# (15 x m)
training_predictions = []

for i in range(0, 6):
    for j in range(i+1, 6):
        model = models_dict[(i, j)]
        training_prediction = model.get_predictions(training_data_x_multi, training_data_y_multi)

        training_answers = []
        
        # Replaces back -1 and 1 with the actual class label 
        for k in training_prediction[0]:
            if k==1:
                training_answers.append(i)
            else:
                training_answers.append(j)

        # appends the predictions made the the (i, j) classifier
        training_predictions.append(training_answers)

training_predictions = np.array(training_predictions)

# contains the final predictions made after computing the class with predicted the most
final_class_train = []

for i in range(training_data_x_multi.shape[1]):
    # iterate over each example
    column_train = training_predictions[:, i]
    classes_train, freq_train = np.unique(column_train, return_counts=True)
    class_train = classes_train[np.argmax(freq_train)]
    final_class_train.append(class_train)

final_class_train = np.array(final_class_train, ndmin=2)

# calculates the accuracy over training dataset
accuracy_multi_cvx_train = np.sum(final_class_train == training_data_y_multi) / training_data_y_multi.shape[1]

# ACCURACY ON VALIDATION DATA
# 2D list containing the predictions over the entire training set made by each classifier
# (15 x n) where n is the number of examples in the validation set
validation_predictions = []

for i in range(0, 6):
    for j in range(i+1, 6):

        model = models_dict[(i, j)]
        
        validation_prediction = model.get_predictions(validation_data_x_multi, validation_data_y_multi)

        validation_answers = []
        
        # Replaces back -1 and 1 with the actual class label 
        for k in validation_prediction[0]:
            if k==1:
                validation_answers.append(i)
            else:
                validation_answers.append(j)

        # appends the predictions made the the (i, j) classifier
        validation_predictions.append(validation_answers)

validation_predictions = np.array(validation_predictions)

# contains the final predictions made after computing the class with predicted the most
final_class_valid = []

for i in range(validation_data_x_multi.shape[1]):
    # iterate over each example
    column_valid = validation_predictions[:, i]
    classes_valid, freq_valid = np.unique(column_valid, return_counts=True)
    class_valid = classes_valid[np.argmax(freq_valid)]
    final_class_valid.append(class_valid)


final_class_valid = np.array(final_class_valid, ndmin=2)

# calculates the accuracy over validation dataset
accuracy_multi_cvx_valid = np.sum(final_class_valid == validation_data_y_multi) / validation_data_y_multi.shape[1]

# CONFUSION MATRIX - CVXOPT
# computes the confusion matrix for the model learnt using CVXOPT

actual_values_cvx = validation_data_y_multi.flatten()           # contains the true labels
predicted_values_cvx = final_class_valid.flatten()              # contains the predicted labels

# 'confusion_matrix' is an in-built function in sklearn.metrics module
confusion_matrix_cvx = confusion_matrix(actual_values_cvx, predicted_values_cvx)

'''
MULTI CLASSIFICATION USING SCIKIT LEARN
GAUSSIAN KERNEL
'''

svm_sklearn_multi = SVM_Sklearn(training_data_x_multi, training_data_y_multi, 'rbf', 1, 0.001)

accuracy_train_multi = svm_sklearn_multi.get_accuracy(training_data_x_multi, training_data_y_multi)               # accuracy over training set
accuracy_validation_multi = svm_sklearn_multi.get_accuracy(validation_data_x_multi, validation_data_y_multi)      # accuracy over validation set

# CONFUSION MATRIX - SKLEARN
# computes the confusion matrix for the model learnt using sklearn

actual_values_sklearn = validation_data_y_multi.flatten()                                               # contains the true labels
predicted_values_sklearn = svm_sklearn_multi.get_predictions(validation_data_x_multi).flatten()         # contains the predicted labels

# 'confusion_matrix' is an in-built function in sklearn.metrics module
confusion_matrix_sklearn = confusion_matrix(actual_values_sklearn, predicted_values_sklearn)

# 12 MIS-LABELLED IMAGES
# shows 12 images which have not been predicted correctly

wrg_img = 0     # we need 12 images
idx = 0
wrg_images = []
while((idx < actual_values_sklearn.shape[0]) and (wrg_img < 12)):
    # if the prediction is wrong, append the image to wrg_images
    if actual_values_sklearn[idx] != predicted_values_sklearn[idx]:
        img_vector = validation_data_x_multi.T[idx]
        wrg_images.append([img_vector, actual_values_sklearn[idx], predicted_values_sklearn[idx]])
        wrg_img += 1
    idx += 1

for v in wrg_images:
    save_image(f'ms-{v[1]}-{v[2]}', v[0])

'''
5-FOLD CROSS-VALIDATION ACCURACY
'''

# calculates the average of the accuracies for the 5 validation subsets formed from the training data for a particular value of C
def train_using_cv(training_data_x, training_data_y, c):
    
    total_train_examples = training_data_x.shape[1]
    batch_size = (int) (total_train_examples/5)             # size of each of the 5 subsets

    accuracy = 0

    for i in range(5):
        valid_start = i * batch_size
        valid_end = valid_start + batch_size

        # for (i)th iteration: validation set is from index 'valid_start' to 'valid_end-1' (both included)
        validation_x = training_data_x[0:768, valid_start:valid_end]        # validation data_x for this iteration
        validation_y = training_data_y[0:768, valid_start:valid_end]        # validation data_y for this iteration

        # concatenates the data before and after the validation subset to get the training data for this iteration
        train_x = np.concatenate((training_data_x[0:768, 0:valid_start], training_data_x[0:768, valid_end:total_train_examples]), axis=1)
        train_y = np.concatenate((training_data_y[0:768, 0:valid_start], training_data_y[0:768, valid_end:total_train_examples]), axis=1)

        model = SVM_Sklearn(train_x, train_y, 'rbf', c, 0.001)
        
        # accuracy over the validation set for this iteration
        curr_accuracy = model.get_accuracy(validation_x, validation_y)

        accuracy += curr_accuracy
    
    # averages the accuracy over 5 iterations
    accuracy /= 5

    return accuracy

# CROSS VALIDATION ACCURACIES
# C = 1e-5
accuracy_1 = train_using_cv(training_data_x_multi, training_data_y_multi, 1e-5)

# C = 1e-3
accuracy_2 = train_using_cv(training_data_x_multi, training_data_y_multi, 1e-3)

# C = 1
accuracy_3 = train_using_cv(training_data_x_multi, training_data_y_multi, 1)

# C = 5
accuracy_4 = train_using_cv(training_data_x_multi, training_data_y_multi, 5)

# C = 10
accuracy_5 = train_using_cv(training_data_x_multi, training_data_y_multi, 10)

# VALIDATION ACCURACIES
# C = 1e-5
model_1 = SVM_Sklearn(training_data_x_multi, training_data_y_multi, 'rbf', 1e-5, 0.001)
accuracy_1_entire = model_1.get_accuracy(validation_data_x_multi, validation_data_y_multi)

# C = 1e-3
model_2 = SVM_Sklearn(training_data_x_multi, training_data_y_multi, 'rbf', 1e-3, 0.001)
accuracy_2_entire = model_2.get_accuracy(validation_data_x_multi, validation_data_y_multi)

# C = 1
model_3 = SVM_Sklearn(training_data_x_multi, training_data_y_multi, 'rbf', 1, 0.001)
accuracy_3_entire = model_3.get_accuracy(validation_data_x_multi, validation_data_y_multi)

# C = 5
model_4 = SVM_Sklearn(training_data_x_multi, training_data_y_multi, 'rbf', 5, 0.001)
accuracy_4_entire = model_4.get_accuracy(validation_data_x_multi, validation_data_y_multi)

# C = 10
model_5 = SVM_Sklearn(training_data_x_multi, training_data_y_multi, 'rbf', 10, 0.001)
accuracy_5_entire = model_5.get_accuracy(validation_data_x_multi, validation_data_y_multi)

# plots the trend in 5-fold cross validation accuracy and accuracy on entire validation set with increasing value of C

c = [1e-5, 1e-3, 1, 5, 10]      # values of C

cv_accuracies = [accuracy_1, accuracy_2, accuracy_3, accuracy_4, accuracy_5]        # 5-fold cross validation accuracies
entire_accuracies = [accuracy_1_entire, accuracy_2_entire, accuracy_3_entire, accuracy_4_entire, accuracy_5_entire]     # accuracies on the entire validation set

plt.xscale('log')       # makes the x-axis log scaled

plt.plot(c, cv_accuracies, label='cross-validation accuracies', color='blue')
plt.plot(c, entire_accuracies, label='validation accuracies', color='red')

plt.xlabel('C')
plt.ylabel('Accuracies')
plt.legend()

plt.show(block=True)