import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Samples a million training samples from Gaussian distributions
# We also add noise to the data - also sampled from a normal distribution
def sample_training_set(number_of_samples, theta_params, gaussian_1, gaussian_2, noise_gaussian):
    mu_1, var_1 = gaussian_1[0], gaussian_1[1]
    mu_2, var_2 = gaussian_2[0], gaussian_2[1]
    mu_noise, var_noise = noise_gaussian[0], noise_gaussian[1]

    std_1, std_2, std_noise = np.sqrt(var_1), np.sqrt(var_2), np.sqrt(var_noise)

    x_1_training_set = np.random.normal(loc=mu_1, scale=std_1, size=(1, number_of_samples))
    x_2_training_set = np.random.normal(loc=mu_2, scale=std_2, size=(1, number_of_samples))
    noise_training_set = np.random.normal(loc=mu_noise, scale=std_noise, size=(1, number_of_samples))

    # Additional row for x_0 feature for each training example
    x_0 = np.ones((1, number_of_samples))

    # Final input and output training dataset before shuffling
    training_set_x = np.vstack((x_0, x_1_training_set, x_2_training_set))
    training_set_y = np.dot(theta_params, training_set_x) + noise_training_set

    # Shuffling the data set to remove any inherent bias
    training_set = np.vstack((training_set_x, training_set_y))
    training_set = training_set.T
    np.random.shuffle(training_set)
    training_set = training_set.T
    # Final input and output training dataset after shuffling
    training_set_x = training_set[0:3, :]
    training_set_y = training_set[3:4, :]

    return (training_set_x, training_set_y)

# Computes the 'least mean square' cost function value for the training set
def get_cost_value(training_set_x, training_set_y, learning_params):
    batch_size = training_set_x.shape[1]
    # print(training_set_x.shape)
    return (1/(2*batch_size)) * np.sum(np.square(training_set_y - np.dot(learning_params, training_set_x)))

# Implementation of the stochastic gradient descent algorithm to learn the theta parameters
def stochastic_gradient_descent(training_set_x, training_set_y, learning_rate, batch_size, allowed_error):
    
    number_of_training_samples = training_set_x.shape[1]
    
    number_of_batches = int (number_of_training_samples/batch_size)
    learning_params = np.zeros((1, 3))
    iteration = 0

    converged = False

    # Stores the values of the cost function of each batch in the last epoch
    batch_cost_values_prev = np.zeros(number_of_batches)
    # Stores the values of the cost function of each batch in the current epoch
    batch_cost_values_new = np.zeros(number_of_batches)
    
    # Stores the values of theta parameters at each iteration
    theta_params_list = np.array([[0, 0, 0]])
    
    while not converged:
        
        # Computes which batch is currently processed (0-indexed)
        batch = iteration % number_of_batches
        
        iteration = iteration + 1

        # Start and end index of training set examples present in the current batch (0-indexed)
        start_idx = (batch) * batch_size
        end_idx = start_idx + batch_size

        # Slices the complete training set to get the batches
        x_batch = training_set_x[:, start_idx : end_idx]
        y_batch = training_set_y[:, start_idx : end_idx]

        # Only during the first epoch, stores the values of cost function.
        # Next epoch onwards, batch_cost_values_prev =  batch_cost_values_new of the last epoch
        if iteration <= number_of_batches:
            batch_cost_values_prev[batch] = get_cost_value(x_batch, y_batch, learning_params)
        
        # Updating theta parameters using gradient descent rule
        h_theta_x = np.dot(learning_params, x_batch)
        loss_array = x_batch * (y_batch - h_theta_x)
        gradient = (1/batch_size) * np.sum(loss_array, axis=1)
        learning_params += learning_rate * gradient

        theta_params_list = np.append(theta_params_list, learning_params, axis=0)
        batch_cost_values_new[batch] = get_cost_value(x_batch, y_batch, learning_params)

        # Check for convergence
        # Happens at the end of each epoch
        # In case the algorithm oscillates about the minima, the convergence is decided by number of iterations.
        if ((iteration % number_of_batches == 0) or (iteration > 100000)):
            
            # Calculates the average of cost values over all the batches for the last and current epoch
            prev_avg = np.average(batch_cost_values_prev)
            new_avg = np.average(batch_cost_values_new)
            if ((abs(new_avg - prev_avg) <= allowed_error) or (iteration > 100000)):
                converged = True
            
            # Updates batch_cost_values_prev
            batch_cost_values_prev = batch_cost_values_new.copy()

    return (learning_params, iteration, theta_params_list.T)

# Function to plot the theta parameters on a 3D graph
def plt_theta_params(theta_0_values, theta_1_values, theta_2_values, plot_axes):
    plot_axes.scatter(theta_0_values, theta_1_values, theta_2_values, c='r', linewidth=0.001)

# Generating the 1 million training examples
number_of_samples = 1000000
actual_theta_params = np.array([[3, 1, 2]])
gaussian_1, gaussian_2, noise_gaussian = [3, 4], [-1, 4], [0, 2]
training_set_x, training_set_y = sample_training_set(number_of_samples, actual_theta_params, gaussian_1, gaussian_2, noise_gaussian)

# Importing the dataset and formatting it
dataframe_x = pd.read_csv('q2test.csv')

test_set = dataframe_x.to_numpy().T

number_of_test_examples  = test_set.shape[1]

x_1_features = test_set[0].reshape((1, number_of_test_examples))
x_2_features = test_set[1].reshape((1, number_of_test_examples))

x_0_features = np.zeros((1, number_of_test_examples))

test_set_x = np.vstack((x_0_features, x_1_features, x_2_features))
test_set_y = test_set[2].reshape((1, number_of_test_examples))

# Training the dataset and computing the cost wrt to training and test dataset
# Batch Size = 1
theta_params_1, iterations_1, theta_params_list_1 = stochastic_gradient_descent(training_set_x, training_set_y, 0.001, 1, 1e-3)

cost_on_training_set_1 = get_cost_value(training_set_x, training_set_y, theta_params_1)
cost_on_test_set_1 = get_cost_value(test_set_x, test_set_y, theta_params_1)

# Plotting the graph
plot_1 = plt.figure(figsize=(10, 10))
plot_1_axes = plot_1.add_axes([0, 0.1, 1, 0.8], projection='3d')
plot_1_axes.set_xlabel('theta_0')
plot_1_axes.set_ylabel('theta_1')
plot_1_axes.set_zlabel('theta_2')
plot_1_axes.set_title('Batch Size = 1')
plt_theta_params(theta_params_list_1[0], theta_params_list_1[1], theta_params_list_1[2], plot_1_axes)


# Batch Size = 100
theta_params_2, iterations_2, theta_params_list_2 = stochastic_gradient_descent(training_set_x, training_set_y, 0.001, 100, 1e-5)

cost_on_training_set_2 = get_cost_value(training_set_x, training_set_y, theta_params_2)
cost_on_test_set_2 = get_cost_value(test_set_x, test_set_y, theta_params_2)

# Plotting the graph
plot_2 = plt.figure(figsize=(10, 10))
plot_2_axes = plot_2.add_axes([0.01, 0.1, 1, 0.8], projection='3d')
plot_2_axes.set_xlabel('theta_0')
plot_2_axes.set_ylabel('theta_1')
plot_2_axes.set_zlabel('theta_2')
plot_2_axes.set_title('Batch Size = 100')
plt_theta_params(theta_params_list_2[0], theta_params_list_2[1], theta_params_list_2[2], plot_2_axes)


# Batch Size = 10000
theta_params_3, iterations_3, theta_params_list_3 = stochastic_gradient_descent(training_set_x, training_set_y, 0.001, 10000, 1e-6)

cost_on_training_set_3 = get_cost_value(training_set_x, training_set_y, theta_params_3)
cost_on_test_set_3 = get_cost_value(test_set_x, test_set_y, theta_params_3)

# Plotting the graph
plot_3 = plt.figure(figsize=(10, 10))
plot_3_axes = plot_3.add_axes([0, 0.1, 1, 0.8], projection='3d')
plot_3_axes.set_xlabel('theta_0')
plot_3_axes.set_ylabel('theta_1')
plot_3_axes.set_zlabel('theta_2')
plot_3_axes.set_title('Batch Size = 10000')
plt_theta_params(theta_params_list_3[0], theta_params_list_3[1], theta_params_list_3[2], plot_3_axes)


# Batch Size = 1000000
theta_params_4, iterations_4, theta_params_list_4 = stochastic_gradient_descent(training_set_x, training_set_y, 0.001, 1000000, 1e-6)

cost_on_training_set_4 = get_cost_value(training_set_x, training_set_y, theta_params_4)
cost_on_test_set_4 = get_cost_value(test_set_x, test_set_y, theta_params_4)

# Plotting the graph
plot_4 = plt.figure(figsize=(10, 10))
plot_4_axes = plot_4.add_axes([0, 0.1, 1, 0.8], projection='3d')
plot_4_axes.set_xlabel('theta_0')
plot_4_axes.set_ylabel('theta_1')
plot_4_axes.set_zlabel('theta_2')
plot_4_axes.set_title('Batch Size = 1000000')
plt_theta_params(theta_params_list_4[0], theta_params_list_4[1], theta_params_list_4[2], plot_4_axes)


# Cost on the test dataset wrt to the actual hypothesis
actual_cost_test_set = get_cost_value(test_set_x, test_set_y, actual_theta_params)

# To ensure that the window showing the plot remains visible
plt.show(block=True)