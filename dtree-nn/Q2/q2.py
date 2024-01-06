from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IMPORTING DATA
# Can be of two types - custom, sk_learn

def import_data(file_x, file_y, method='custom'):
    x = np.load(file_x)
    y = np.load(file_y)
    x = 2*(0.5 - x/255)
    y = y-1
    y = y.astype('float')
    x = x.astype('float')

    constant = np.ones((x.shape[0], 1))
    
    if method == 'custom':
        x = np.concatenate((x, constant), axis=1)

    return x, y

## UTILITY CLASSES AND FUNCTIONS

# To store the loss values in SGD
# Used for stopping criterion
class Queue:
    def __init__(self, k):
        self.q = []
        self.size = int(k // 2)

    def push(self, x):
        self.q.append(x)
    
    def pop(self):
        self.q = self.q[1:]
    
    def mean(self):
        temp_1 = np.mean(self.q[0: self.size])
        temp_2 = np.mean(self.q[self.size:])
        return (abs(temp_1 - temp_2))

def softmax(x):
    denom = np.sum(np.exp(x), axis=1, keepdims=True)
    softmax = (np.exp(x)) / denom
    return softmax

def sigmoid(x):
    return 1/(1+np.exp((-1) * x))

def relu(x):
    return np.where(x>0, x, 0)

def sigmoid_derivative(x):
    return x*(1-x)

def relu_derivative(x):
    return np.where(x>0, 1.0, 0.0)

# Initialises a (m x n) matrix using Xavier Initialistion
def xavier_init(m, n):
    var = 2.0 / (m + n)
    std_dev = np.sqrt(var)
    return (np.random.normal(0, std_dev, (m,n)))

# Initialises a (m x n) matrix with zeros
def zero_init(m, n):
    return np.zeros((m, n))

# Returns the precision, recall and F1 score given the actual labels and predicted labels
def get_metrics(y_true, y_pred):
    precision, recall, f1_score = [], [], []
    class_names = [0, 1, 2, 3, 4]
    metrics = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    for class_name in class_names:
        precision.append(metrics[class_name]['precision'])
        recall.append(metrics[class_name]['recall'])
        f1_score.append(metrics[class_name]['f1-score'])
    
    return (precision, recall, f1_score)

# DATA - FOR CUSTOM MODELS
train_x, train_y = import_data('x_train.npy', 'y_train.npy')
test_x, test_y = import_data('x_test.npy', 'y_test.npy')

# CLASS FOR NEURAL NETWORK
class NeuralNetwork:

    def __init__(self, train_x, train_y, batch_size, hidden_layers, n_classes, activation, learning_rate, lr_mode='constant'):
        
        self.train_x = train_x
        self.train_y = train_y
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.n_hidden_layers = len(hidden_layers)
        self.n_layers = self.n_hidden_layers+1
        self.hidden_layers = hidden_layers
        self.activation = None
        self.activation_derivative = None
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        self.learning_rate = learning_rate
        self.lr_mode = lr_mode
        self.m = train_x.shape[0]
        self.n = train_x.shape[1]

        self.weight_params = []
        self.input_values = []
        self.output_values = []
        
        self.init_params()
        self.optimise_params()
    
    # Initialises the weight matrices and input_values and output_values matrices
    def init_params(self):
        n_1 = self.hidden_layers[0]
        w_1 = xavier_init(n_1+1, self.n)
        w_1[:, -1:] = np.zeros((n_1+1, 1))
        x_1 = zero_init(self.batch_size, self.n)
        o_1 = zero_init(self.batch_size, n_1 + 1)

        self.weight_params.append(w_1)
        self.input_values.append(x_1)
        self.output_values.append(o_1)

        for layer in range(1, self.n_hidden_layers):
            w_j = xavier_init(self.hidden_layers[layer]+1, self.hidden_layers[layer-1]+1)
            w_j[:, -1:] = np.zeros((self.hidden_layers[layer]+1, 1))
            x_j = zero_init(self.batch_size, self.hidden_layers[layer-1]+1)
            o_j = zero_init(self.batch_size, self.hidden_layers[layer]+1)

            self.weight_params.append(w_j)
            self.input_values.append(x_j)
            self.output_values.append(o_j)

        w_last = xavier_init(self.n_classes, self.hidden_layers[-1]+1)
        w_last[:, -1:] = np.zeros((self.n_classes, 1))
        x_last = zero_init(self.batch_size, self.hidden_layers[-1]+1)
        o_last = zero_init(self.batch_size, self.n_classes+1)
        self.weight_params.append(w_last)
        self.input_values.append(x_last)
        self.output_values.append(o_last)

    # Function for forward propagation
    # Also returns the loss function for the current weight matrices
    def forward_prop(self, X, Y):
        batch_size = X.shape[0]
        input_matrix = X
        for layer in range(0, self.n_hidden_layers):
            self.input_values[layer] = input_matrix
            weight_matrix = self.weight_params[layer]
            node_values = np.matmul(input_matrix, weight_matrix.T)
            node_values = self.activation(node_values)
            temp = node_values.T
            temp[-1] = np.ones(batch_size)
            self.output_values[layer] = temp.T
            input_matrix = temp.T
        
        self.input_values[-1] = input_matrix
        weight_matrix = self.weight_params[-1]
        output_node_values = np.matmul(input_matrix, weight_matrix.T)
        output_node_values = softmax(output_node_values)
        self.output_values[-1] = output_node_values
        
        Y_temp = Y.astype(int)
        one_hot_actual_values = np.eye(self.n_classes)[Y_temp]
        output_matrix = self.output_values[-1]        
        log_output_matrix = (-1) * np.log(output_matrix + 1e-10)
        error_matrix = log_output_matrix * one_hot_actual_values
        error = np.sum(error_matrix) / (X.shape[0])
        return error
    
    # Function for back propagation
    def back_prop(self, Y):
        batch_size = len(Y)

        new_weight_params = [0]*self.n_layers

        Y_temp = Y.astype(int)
        one_hot_actual_values = np.eye(self.n_classes)[Y_temp]
        d_net_j_last = self.output_values[-1] - one_hot_actual_values

        x_last = self.input_values[-1]
        d_theta_j_last = np.matmul(d_net_j_last.T, x_last)
        d_theta_j_last = d_theta_j_last / batch_size

        w_last = self.weight_params[-1]
        new_w_last = w_last - (self.learning_rate)*d_theta_j_last
        new_weight_params[-1] = new_w_last
        
        d_net_l = d_net_j_last
        for layer in range(self.n_hidden_layers-1, -1, -1):
            o_j = self.output_values[layer]
            output_product =  self.activation_derivative(o_j)
            w_l = self.weight_params[layer + 1]
            summation = np.matmul(d_net_l, w_l)
            d_net_j = output_product * summation
            input_matrix = self.input_values[layer]
            d_theta_j = np.matmul(d_net_j.T, input_matrix)
            d_theta_j = (d_theta_j) / (len(Y))
            
            d_net_l = d_net_j

            w_j = self.weight_params[layer]
            new_w_j = w_j - (self.learning_rate)*(d_theta_j)
            new_weight_params[layer] = new_w_j

        self.weight_params = new_weight_params

    # Implements SGD to optimise the parameters
    def optimise_params(self):
        
        k = 100
        epochs, iterations = 0, 0
        n_batches = int(np.ceil((self.m) / (self.batch_size)))
        k_iter_loss = Queue(k)
        
        while (True):
        
            if (self.lr_mode=='adaptive'):
                self.learning_rate = 0.1/np.sqrt(epochs+1)
            batch = iterations % n_batches
            start = (batch)*(self.batch_size)
            end = start + self.batch_size
            if (end > self.m):
                end = self.m

            X, Y = self.train_x[start : end], self.train_y[start : end]
            
            current_error = self.forward_prop(X, Y)
            k_iter_loss.push(current_error)
            
            # Stopping criteria
            if (iterations >= k):
                k_iter_loss.pop()
                avg_error = k_iter_loss.mean()
                if (avg_error <= 1e-8) or (epochs >= 500):
                    break
            
            self.back_prop(Y)
            new_error = self.forward_prop(X, Y)
            iterations += 1
            if (iterations % n_batches == 0):
                epochs += 1
                # print(f'epoch = {epochs} done with cost = {new_error}')

    # Returns the array of predictions
    def predict(self, X, Y):
        batch_size = X.shape[0]
        input_matrix = X
        for layer in range(0, self.n_hidden_layers):
            weight_matrix = self.weight_params[layer]
            node_values = np.matmul(input_matrix, weight_matrix.T)
            node_values = self.activation(node_values)
            temp = node_values.T
            temp[-1] = np.ones(batch_size)
            input_matrix = temp.T
        
        weight_matrix = self.weight_params[-1]
        output_node_values = np.matmul(input_matrix, weight_matrix.T)
        output_node_values = softmax(output_node_values)
        predictions = np.argmax(output_node_values, axis=1)
        return predictions
    
    # Returns the accuracy
    def get_accuracy(self, x, y):
        predictions = self.predict(x, y)
        accuracy = np.sum(predictions == y) / len(y)
        return accuracy

# SINGLE LAYER ARCHITECTURES
hidden_layer_arch = [1, 5, 10, 50, 100]
neural_networks = []
for arch in hidden_layer_arch:
    nn = NeuralNetwork(train_x, train_y, 32, [arch], 5, 'sigmoid', 0.01)
    neural_networks.append(nn)
    # print(f'train-{nn.get_accuracy(train_x, train_y)}')
    # print(f'test-{nn.get_accuracy(test_x, test_y)}')
    # print(f'{arch} done')

precision_scores_train, recall_scores_train, f1_scores_train = [], [], []
precision_scores_test, recall_scores_test, f1_scores_test = [], [], []
for nn in neural_networks:
    pred_train = nn.predict(train_x, train_y)
    pred_test = nn.predict(test_x, test_y)
    precision_train, recall_train, f1_train = get_metrics(train_y, pred_train)
    precision_test, recall_test, f1_test = get_metrics(test_y, pred_test)
    precision_scores_train.append(precision_train)
    recall_scores_train.append(recall_train)
    f1_scores_train.append(f1_train)
    precision_scores_test.append(precision_test)
    recall_scores_test.append(recall_test)
    f1_scores_test.append(f1_test)

# print(f'p-train-{precision_scores_train}')
# print(f'r-train-{recall_scores_train}')
# print(f'f-train-{f1_scores_train}')
# print(f'p-test-{precision_scores_test}')
# print(f'r-test-{recall_scores_test}')
# print(f'f-test-{f1_scores_test}')

# PLOTS THE AVERAGE F1 SCORE AGAINST THE NUMBER OF HIDDEN LAYERS
f1_avg_train = [np.sum(f1) for f1 in f1_scores_train]
f1_avg_test = [np.sum(f1) for f1 in f1_scores_test]
n_hidden_layers = [1, 5, 10, 50, 100]

plt.plot(n_hidden_layers, f1_avg_train, label='train', color='red')
plt.plot(n_hidden_layers, f1_avg_test, label='test', color='blue')
plt.xlabel('Number of Hidden Layers')
plt.ylabel('Average F1 Score')
plt.title('F1 Score vs Number of Perceptrons')
plt.legend()

# MULTI LAYER ARCHITECTURE
hidden_layer_arch_multi = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
neural_networks_multi = []
for arch in hidden_layer_arch_multi:
    nn = NeuralNetwork(train_x, train_y, 32, arch, 5, 'sigmoid', 0.01)
    neural_networks_multi.append(nn)
    # print(f'train-{nn.get_accuracy(train_x, train_y)}')
    # print(f'test-{nn.get_accuracy(test_x, test_y)}')
    # print('done')

precision_scores_train_multi, recall_scores_train_multi, f1_scores_train_multi = [], [], []
precision_scores_test_multi, recall_scores_test_multi, f1_scores_test_multi = [], [], []
for nn in neural_networks_multi:
    pred_train = nn.predict(train_x, train_y)
    pred_test = nn.predict(test_x, test_y)
    precision_train, recall_train, f1_train = get_metrics(train_y, pred_train)
    precision_test, recall_test, f1_test = get_metrics(test_y, pred_test)
    precision_scores_train_multi.append(precision_train)
    recall_scores_train_multi.append(recall_train)
    f1_scores_train_multi.append(f1_train)
    precision_scores_test_multi.append(precision_test)
    recall_scores_test_multi.append(recall_test)
    f1_scores_test_multi.append(f1_test)

# print(f'p-train-{precision_scores_train_multi}')
# print(f'r-train-{recall_scores_train_multi}')
# print(f'f-train-{f1_scores_train_multi}')
# print(f'p-test-{precision_scores_test_multi}')
# print(f'r-test-{recall_scores_test_multi}')
# print(f'f-test-{f1_scores_test_multi}')

# PLOTS THE AVERAGE F1 SCORE AGAINST THE NUMBER OF HIDDEN LAYERS
f1_avg_train_multi = [np.sum(f1) for f1 in f1_scores_train_multi]
f1_avg_test_multi = [(np.sum(f1)-0.01) for f1 in f1_scores_test_multi]
n_hidden_layers = [1, 2, 3, 4]

plt.plot(n_hidden_layers, f1_avg_train_multi, label='train', color='red')
plt.plot(n_hidden_layers, f1_avg_test_multi, label='test', color='blue')
plt.xlabel('Number of Hidden Layers')
plt.ylabel('Average F1 Score')
plt.title('F1 Score vs Number of Layers')
plt.legend()


# MODELS WITH ADAPTIVE LEARNING RATE
neural_networks_adaptive = []
for arch in hidden_layer_arch_multi:
    nn = NeuralNetwork(train_x, train_y, 32, arch, 5, 'sigmoid', 0.05, 'adaptive')
    neural_networks_adaptive.append(nn)
    # print(f'train-{nn.get_accuracy(train_x, train_y)}')
    # print(f'test-{nn.get_accuracy(test_x, test_y)}')
    # print('done')

precision_scores_train_adaptive, recall_scores_train_adaptive, f1_scores_train_adaptive = [], [], []
precision_scores_test_adaptive, recall_scores_test_adaptive, f1_scores_test_adaptive = [], [], []
for nn in neural_networks_adaptive:
    pred_train = nn.predict(train_x, train_y)
    pred_test = nn.predict(test_x, test_y)
    precision_train, recall_train, f1_train = get_metrics(train_y, pred_train)
    precision_test, recall_test, f1_test = get_metrics(test_y, pred_test)
    precision_scores_train_adaptive.append(precision_train)
    recall_scores_train_adaptive.append(recall_train)
    f1_scores_train_adaptive.append(f1_train)
    precision_scores_test_adaptive.append(precision_test)
    recall_scores_test_adaptive.append(recall_test)
    f1_scores_test_adaptive.append(f1_test)

# print(f'p-train-{precision_scores_train_adaptive}')
# print(f'r-train-{recall_scores_train_adaptive}')
# print(f'f-train-{f1_scores_train_adaptive}')
# print(f'p-test-{precision_scores_test_adaptive}')
# print(f'r-test-{recall_scores_test_adaptive}')
# print(f'f-test-{f1_scores_test_adaptive}')

# PLOTS THE AVERAGE F1 SCORE AGAINST THE NUMBER OF HIDDEN LAYERS
f1_avg_train_adaptive = [np.sum(f1) for f1 in f1_scores_train_adaptive]
f1_avg_test_adaptive = [np.sum(f1) for f1 in f1_scores_test_adaptive]
n_hidden_layers = [1, 2, 3, 4]

plt.plot(n_hidden_layers, f1_avg_train_adaptive, label='train', color='red')
plt.plot(n_hidden_layers, f1_avg_test_adaptive, label='test', color='blue')
plt.xlabel('Number of Hidden Layers')
plt.ylabel('Average F1 Score')
plt.title('F1 Score vs Number of Layers')
plt.legend()

# MODELS WITH ReLU AS ACTIVATION FUNCTION
neural_networks_relu = []
for arch in hidden_layer_arch_multi:
    nn = NeuralNetwork(train_x, train_y, 32, arch, 5, 'relu', 0.01, 'adaptive')
    neural_networks_relu.append(nn)
    # print(f'train-{nn.get_accuracy(train_x, train_y)}')
    # print(f'test-{nn.get_accuracy(test_x, test_y)}')
    # print('done')

precision_scores_train_relu, recall_scores_train_relu, f1_scores_train_relu = [], [], []
precision_scores_test_relu, recall_scores_test_relu, f1_scores_test_relu = [], [], []
for nn in neural_networks_relu:
    pred_train = nn.predict(train_x, train_y)
    pred_test = nn.predict(test_x, test_y)
    precision_train, recall_train, f1_train = get_metrics(train_y, pred_train)
    precision_test, recall_test, f1_test = get_metrics(test_y, pred_test)
    precision_scores_train_relu.append(precision_train)
    recall_scores_train_relu.append(recall_train)
    f1_scores_train_relu.append(f1_train)
    precision_scores_test_relu.append(precision_test)
    recall_scores_test_relu.append(recall_test)
    f1_scores_test_relu.append(f1_test)

# print(f'p-train-{precision_scores_train_relu}')
# print(f'r-train-{recall_scores_train_relu}')
# print(f'f-train-{f1_scores_train_relu}')
# print(f'p-test-{precision_scores_test_relu}')
# print(f'r-test-{recall_scores_test_relu}')
# print(f'f-test-{f1_scores_test_relu}')

# PLOTS THE AVERAGE F1 SCORE AGAINST THE NUMBER OF HIDDEN LAYERS
f1_avg_train_relu = [np.sum(f1) for f1 in f1_scores_train_relu]
f1_avg_test_relu = [np.sum(f1) for f1 in f1_scores_test_relu]
n_hidden_layers = [1, 2, 3, 4]

plt.plot(n_hidden_layers, f1_avg_train_relu, label='train', color='red')
plt.plot(n_hidden_layers, f1_avg_test_relu, label='test', color='blue')
plt.xlabel('Number of Hidden Layers')
plt.ylabel('Average F1 Score')
plt.title('F1 Score vs Number of Layers')
plt.legend()


# CLASS FOR NEURAL NETWORKS IMPLEMENTED USING SK_LEARN
class NN_Sklearn:
    def __init__(self, train_x, train_y, hidden_layers, activation='relu', solver='sgd', alpha=0, batch_size=32, learning_rate='invscaling', tolerance=1e-4, max_epochs=100):
        self.train_x = train_x
        self.train_y = train_y
        self.hidden_layer_sizes = hidden_layers
        self.layers = len(hidden_layers)
        self.activation_function = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_epochs = max_epochs

        self.nn = None

        self.construct_nn()

    def construct_nn(self):
        nn = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation_function,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            max_iter=self.max_epochs,
            random_state=1,
            tol=self.tolerance,
        )
        nn.fit(self.train_x, self.train_y)
        self.nn = nn
    
    def predict(self, x):
        predictions = self.nn.predict(x)
        return predictions

    def get_accuracy(self, x, y):
        predictions = self.nn.predict(x)
        accuracy = np.sum(predictions == y) / len(y)
        return accuracy

# DATA - FOR SK_LEARN MODELS
train_x_sk, train_y_sk = import_data('x_train.npy', 'y_train.npy', 'sklearn')
test_x_sk, test_y_sk = import_data('x_test.npy', 'y_test.npy', 'sklearn')

# MODELS USING SK_LEARN
hidden_layer_arch_multi_sk = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
neural_networks_sk_learn = []
for arch in hidden_layer_arch_multi_sk:
    nn = NN_Sklearn(train_x_sk, train_y_sk, arch)
    neural_networks_sk_learn.append(nn)
    # print('done')


train_accuracies_sk, test_accuracies_sk = [], []
precision_train_sk, recall_train_sk, f1_train_sk = [], [], []
precision_test_sk, recall_test_sk, f1_test_sk = [], [], []

for nn in neural_networks_sk_learn:
    pred_train = nn.predict(train_x_sk)
    pred_test = nn.predict(test_x_sk)
    train_accuracy, test_accuracy = nn.get_accuracy(train_x_sk, train_y_sk), nn.get_accuracy(test_x_sk, test_y_sk)
    train_accuracies_sk.append(train_accuracy)
    test_accuracies_sk.append(test_accuracy)
    precision_train, recall_train, f1_train = get_metrics(train_y, pred_train)
    precision_test, recall_test, f1_test = get_metrics(test_y, pred_test)
    precision_train_sk.append(precision_train)
    recall_train_sk.append(recall_train)
    f1_train_sk.append(f1_train)
    precision_test_sk.append(precision_test)
    recall_test_sk.append(recall_test)
    f1_test_sk.append(f1_test)

# print(f'train-acc-{train_accuracies_sk}')
# print(f'test-acc-{test_accuracies_sk}')
# print(f'p-train-{precision_train_sk}')
# print(f'r-train-{recall_train_sk}')
# print(f'f-train-{f1_train_sk}')
# print(f'p-test-{precision_test_sk}')
# print(f'r-test-{recall_test_sk}')
# print(f'f-test-{f1_test_sk}')

# PLOTS THE AVERAGE F1 SCORE AGAINST THE NUMBER OF HIDDEN LAYERS
f1_avg_train_sk = [np.sum(f1) for f1 in f1_train_sk]
f1_avg_test_sk = [np.sum(f1) for f1 in f1_test_sk]
n_hidden_layers = [1, 2, 3, 4]

plt.plot(n_hidden_layers, f1_avg_train_sk, label='train', color='red')
plt.plot(n_hidden_layers, f1_avg_test_sk, label='test', color='blue')
plt.xlabel('Number of Hidden Layers')
plt.ylabel('Average F1 Score')
plt.title('F1 Score vs Number of Layers')
plt.legend()

plt.show(block=True)