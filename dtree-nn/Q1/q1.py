from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# IMPORT AND ENCODE DATA
# POSSIBLE ENCODERS - ordinal_encoder, one-hot_encoder
def get_encoded_data(file_name, encoder):
    data = pd.read_csv(file_name)
    examples = data.shape[0]
    
    encoded_labels = ['team','host','opp','month', 'day_match']
    remaining_labels = ["year","toss","bat_first","format" ,"fow","score" ,"rpo" ,"result"]
    
    label_encoder = None
    
    if encoder == 'ordinal_encoder':
        label_encoder = OrdinalEncoder()
    elif encoder == 'one-hot_encoder':
        label_encoder = OneHotEncoder(sparse_output = False)
        
    label_encoder.fit(data[encoded_labels])
    
    data_1 = pd.DataFrame(label_encoder.transform(data[encoded_labels]), columns = label_encoder.get_feature_names_out()).to_numpy()
    data_2 = data[remaining_labels].to_numpy()

    temp = np.full((examples, 2), 0.0)
    
    final_data = []
    if encoder == 'ordinal_encoder':
        final_data = np.concatenate((data_1, data_2), axis=1)
    elif encoder == 'one-hot_encoder':
        # adds two features for the values which were not included in the training data
        final_data = np.concatenate((data_1, temp, data_2), axis=1)
    
    X = final_data[:, :-1]
    Y = final_data[:, -1:]
    
    return X, Y.flatten()

# GET FEATURE INFORMATION <br>
# Feature type and number of categories <br>
# Generates an array of which features are available for split
def get_feature_info(encoder):
    
    feature_type, feature_ct, feature_allowed = [], [], []

    if encoder == 'ordinal_encoder':
        feature_type = ['cat', 'cat', 'cat', 'cat', 'cat', 'cont', 'cat', 'cat', 'cat', 'cont', 'cont', 'cont']
        feature_ct = [20, 19, 20, 12, 3, -1, 2, 2, 2, -1, -1, -1]
        feature_allowed = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    elif encoder == 'one-hot_encoder':
        encoded_ct = 74
        feature_type = np.full((encoded_ct, ), 'cat')
        feature_ct = np.full((encoded_ct, ), 2)

        feature_temp_type = np.array(['cont', 'cat', 'cat', 'cat', 'cont', 'cont', 'cont'])
        feature_temp_ct = np.array([-1, 2, 2, 2, -1, -1, -1])

        feature_type = np.concatenate((feature_type, feature_temp_type))
        feature_ct = np.concatenate((feature_ct, feature_temp_ct))
        feature_allowed = np.full((feature_type.size, ), 1)

    return feature_type, feature_ct, feature_allowed

# DATA - ORDINAL ENCODER
train_x_ordinal, train_y_ordinal = get_encoded_data('train.csv', 'ordinal_encoder')
validation_x_ordinal, validation_y_ordinal = get_encoded_data('val.csv', 'ordinal_encoder')
test_x_ordinal, test_y_ordinal = get_encoded_data('test.csv', 'ordinal_encoder')

# DATA - ONE-HOT ENCODER
train_x_one_hot, train_y_one_hot = get_encoded_data('train.csv', 'one-hot_encoder')
validation_x_one_hot, validation_y_one_hot = get_encoded_data('val.csv', 'one-hot_encoder')
test_x_one_hot, test_y_one_hot = get_encoded_data('test.csv', 'one-hot_encoder')

# CLASS FOR DECISION TREE NODE
class DTNode:
    def __init__(self):
        self.id = 0
        self.children = []
        self.subtree_nodes = 1
        self.subtree_leaf_nodes = 0
        self.is_leaf = True
        self.label_leaf = 0
        self.label_if_leaf = 0
        self.split_attr = None
        self.median_value = 0

# CLASS FOR DECISION TREE
class DTree:
    def __init__(self, train_x, train_y, max_depth, encoder):
        
        self.root = DTNode()
        
        self.max_depth = max_depth
        
        self.train_x = train_x
        self.train_y = train_y
        self.feature_type, self.feature_ct, self.feature_allowed = get_feature_info(encoder)
        
        self.nodes = 0
        self.leaf_nodes = 0

        # For Pruning
        self.node_to_prune = None
        self.best_accuracy = -1
        self.ct = 0
        self.make_decision_tree()

    # Computes the entropy 
    def get_entropy(self, y):
        zeros, ones = len(y[y==0]), len(y[y==1])
        if (zeros==0 or ones==0):
            return 0
        zero_prob, one_prob = zeros/(ones+zeros), ones/(ones+zeros)
        entropy = -(zero_prob * np.log(zero_prob) + one_prob*np.log(one_prob))
        return entropy

    # Computes the conditional entropy given an attribute
    def get_cond_entropy(self, x, y, feature):
        feature_row = x.T[feature]
        entropy = 0

        if self.feature_type[feature] == 'cat':
            categories = np.unique(feature_row)
            for category in categories:
                prob_category = np.sum(feature_row == category) / len(feature_row)
                entropy_given_category = self.get_entropy(y[feature_row == category])
                entropy += prob_category * entropy_given_category
        
        elif self.feature_type[feature] == 'cont':
            median = np.median(feature_row)
            # less than equal to
            prob_less = np.sum(feature_row <= median) / len(feature_row)
            entropy_given_less = self.get_entropy(y[feature_row <= median])
            entropy += prob_less * entropy_given_less
            # greater than
            prob_more = np.sum(feature_row > median) / len(feature_row)
            entropy_given_more = self.get_entropy(y[feature_row > median])
            entropy += prob_more * entropy_given_more
        
        return entropy
    
    # Computes the mutual information
    def get_mutual_information(self, x, y, feature):
        entropy = self.get_entropy(y)
        cond_entropy = self.get_cond_entropy(x, y, feature)
        return (entropy - cond_entropy)
    
    # Returns the idx of the best attribute to split on
    def choose_best_attribute(self, x, y):
        best_attribute = -1
        max_mutual_information = -1
        columns = x.shape[1]
        for feature in range(columns):
            if self.feature_allowed[feature] == 1:
                mutual_information = self.get_mutual_information(x, y, feature)
                if mutual_information > max_mutual_information:
                    max_mutual_information = mutual_information
                    best_attribute = feature
        return best_attribute
    
    # Splits the data into categories given an attribute
    def split_data(self, x, y, feature):
        data_x, data_y = [], []
        feature_row = x.T[feature]
        if self.feature_type[feature] == 'cat':
            for i in range(self.feature_ct[feature]):
                x_subset = x[feature_row == i]
                y_subset = y[feature_row == i]
                data_x.append(x_subset)
                data_y.append(y_subset)
        elif self.feature_type[feature] == 'cont':
            median = np.median(feature_row)
            x_less, y_less = x[feature_row <= median], y[feature_row <= median]
            x_more, y_more = x[feature_row > median], y[feature_row > median]
            data_x.append(x_less)
            data_x.append(x_more)
            data_y.append(y_less)
            data_y.append(y_more)

        return (data_x, data_y)
    
    # Returns 1 or 0 depending on which label occurs more in y
    def get_max_label(self, y):
        ones = y[y==1].size
        zeros = y[y==0].size
        if ones > zeros:
            return 1
        else:
            return 0

    # Recursively grows the decision tree
    def grow_decision_tree(self, node, x, y, depth_allowed):
        
        node.id = self.ct
        self.ct += 1
        
        self.nodes += 1
        
        label_if_leaf = self.get_max_label(y)
        node.label_if_leaf = label_if_leaf
                
        if (depth_allowed == 0):
            self.leaf_nodes += 1
            # node.subtree_nodes += 1
            # node.subtree_leaf_nodes += 1
            node.is_leaf = True
            node.label_leaf = label_if_leaf
            return
        
        if (np.all(y==0) or np.all(y==1)):
            self.leaf_nodes += 1
            # node.subtree_nodes += 1
            # node.subtree_leaf_nodes += 1
            node.is_leaf = True
            node.label_leaf = y[0]
            return

        node.is_leaf = False
        
        split_attr = self.choose_best_attribute(x, y)
        
        if (split_attr == -1):
            self.leaf_nodes += 1
            # node.subtree_nodes += 1
            # node.subtree_leaf_nodes += 1
            node.is_leaf = True
            node.label_leaf = label_if_leaf
            return
        
        node.split_attr = split_attr

        if self.feature_type[split_attr] == 'cat':
            self.feature_allowed[split_attr] = 0
            
        if self.feature_type[split_attr] == 'cont':
            feature_row = x.T[split_attr]
            median = np.median(feature_row)
            node.median_value = median

        data_x_list, data_y_list = self.split_data(x, y, split_attr)
        children = len(data_x_list)
        for i in range(children):
            child = DTNode()
            node.children.append(child)
            if (data_y_list[i].size == 0):
                self.nodes += 1
                self.leaf_nodes += 1
                node.subtree_nodes += 1
                node.subtree_leaf_nodes += 1
                child.is_leaf = True
                child.label_leaf = label_if_leaf
            else:
                self.grow_decision_tree(child, data_x_list[i], data_y_list[i], depth_allowed-1)
                node.subtree_nodes += child.subtree_nodes
                node.subtree_leaf_nodes += child.subtree_leaf_nodes
            self.feature_allowed[split_attr] = 1

    # Entry function to construct the decision tree
    def make_decision_tree(self):
        self.grow_decision_tree(self.root, self.train_x, self.train_y, self.max_depth)
        self.ct = 0
        
    # Traverses the tree to predict the label given examples x
    def traverse_tree(self, node, x):
        if node.is_leaf == True:
            return node.label_leaf
        
        feature = node.split_attr
        if self.feature_type[feature] == 'cat':
            current_value = int(x[feature])
            child = node.children[current_value]
            return self.traverse_tree(child, x)
        elif self.feature_type[feature] == 'cont':
            current_value = x[feature]
            split_val = node.median_value
            child = None
            if current_value <= split_val:
                child = node.children[0]
            else:
                child = node.children[1]
            return self.traverse_tree(child, x)
        
    # Selects the node which on pruning gives the best validation accuracy
    def select_node_to_prune(self, node, x_val, y_val):
        if (node.is_leaf == True):
            return
        node.is_leaf = True
        node.label_leaf = node.label_if_leaf
        new_accuracy = self.get_accuracy(x_val, y_val)
        if new_accuracy > self.best_accuracy:
            self.best_accuracy = new_accuracy
            self.node_to_prune = node
        node.is_leaf = False
        for child in node.children:
            self.select_node_to_prune(child, x_val, y_val)
        return
    
    # Decides whether to prune the node returned by select_node_to_prune or not
    def prune(self, x_val, y_val, curr_accuracy):      
        self.best_accuracy = -1
        self.select_node_to_prune(self.root, x_val, y_val)
        if (self.best_accuracy > curr_accuracy):
            self.node_to_prune.is_leaf = True
            self.node_to_prune.label_leaf = self.node_to_prune.label_if_leaf
            self.nodes = (self.nodes) - (self.node_to_prune.subtree_nodes) + 1
            self.leaf_nodes = (self.leaf_nodes) - (self.node_to_prune.subtree_leaf_nodes) + 1

    # Entry function to start post-pruning the decision tree
    def post_prune(self, x_val, y_val, x_train, y_train, x_test, y_test):
        train_accuracy, test_accuracy, val_accuracy = [], [], []
        nodes = []

        while(True):
            prev_val_accuracy = self.get_accuracy(x_val, y_val)
            
            train_accuracy.append(self.get_accuracy(x_train, y_train))
            test_accuracy.append(self.get_accuracy(x_test, y_test))
            val_accuracy.append(prev_val_accuracy)
            nodes.append(self.nodes)
            
            self.prune(x_val, y_val, prev_val_accuracy)
            
            new_val_accuracy = self.get_accuracy(x_val, y_val)
            
            if ((new_val_accuracy - prev_val_accuracy) <= 1e-5):
                train_accuracy.append(self.get_accuracy(x_train, y_train))
                test_accuracy.append(self.get_accuracy(x_test, y_test))
                val_accuracy.append(new_val_accuracy)
                nodes.append(self.nodes)
                break
        
        return (train_accuracy, val_accuracy, test_accuracy, nodes)

    # Returns the prediction accuracy
    def get_accuracy(self, x, y):
        predictions = []
        for i in range(x.shape[0]):
            prediction = self.traverse_tree(self.root, x[i])
            predictions.append(prediction)
        predictions = np.array(predictions)
        accuracy = np.sum(predictions == y) / len(predictions)
        return accuracy

# DTREE MODELS
# Ordinal Encoding 
max_depths_ordinal = [5, 10, 15, 20, 25]
d_trees_ordinal = []
train_accuracies_ordinal, test_accuracies_ordinal = [], []

for max_depth in max_depths_ordinal:
    d_tree = DTree(train_x_ordinal, train_y_ordinal, max_depth, 'ordinal_encoder')
    train_accuracy = d_tree.get_accuracy(train_x_ordinal, train_y_ordinal)
    test_accuracy = d_tree.get_accuracy(test_x_ordinal, test_y_ordinal)
    d_trees_ordinal.append(d_tree)
    train_accuracies_ordinal.append(train_accuracy)
    test_accuracies_ordinal.append(test_accuracy)

Y_only_win, Y_only_lose = np.ones(len(test_y_ordinal)), np.zeros(len(test_y_ordinal))
only_win_accuracy = np.sum(Y_only_win == test_y_ordinal) / len(test_y_ordinal)
only_lose_accuracy = np.sum(Y_only_lose == test_y_ordinal) / len(test_y_ordinal)

# print(f'train - {train_accuracies_ordinal}')
# print(f'test - {test_accuracies_ordinal}')
# print(f'win - {only_win_accuracy}')
# print(f'lose - {only_lose_accuracy}')

# PLOT FOR ACCURACIES VS DEPTH - ORDINAL ENCODING
plt.plot(max_depths_ordinal, train_accuracies_ordinal, label='train', color='red')
plt.plot(max_depths_ordinal, test_accuracies_ordinal, label='test', color='blue')
plt.xlabel('Max Depths')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs Max Depth for Ordinal Encoding')

# DTREE MODELS
# One-Hot Encoding
max_depths_one_hot = [15, 25, 35, 45]
d_trees_one_hot = []
train_accuracies_one_hot, test_accuracies_one_hot = [], []

for max_depth in max_depths_one_hot:
    d_tree = DTree(train_x_one_hot, train_y_one_hot, max_depth, 'one-hot_encoder')
    train_accuracy = d_tree.get_accuracy(train_x_one_hot, train_y_one_hot)
    test_accuracy = d_tree.get_accuracy(test_x_one_hot, test_y_one_hot)
    d_trees_one_hot.append(d_tree)
    train_accuracies_one_hot.append(train_accuracy)
    test_accuracies_one_hot.append(test_accuracy)

# print(f'train - {train_accuracies_one_hot}')
# print(f'test - {test_accuracies_one_hot}')

# PLOT FOR ACCURACIES VS DEPTH - ONE-HOT ENCODING
plt.plot(max_depths_one_hot, train_accuracies_one_hot, label='train', color='red')
plt.plot(max_depths_one_hot, test_accuracies_one_hot, label='test', color='blue')
plt.xlabel('Max Depths')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs Max Depth for One-Hot Encoding')

# DTREE MODELS
# With Pruning
train_accuracies_list, val_accuracies_list, test_accuracies_list, nodes_list = [], [], [], []
for d_tree in d_trees_one_hot:
    result = d_tree.post_prune(validation_x_one_hot, validation_y_one_hot, train_x_one_hot, train_y_one_hot, test_x_one_hot, test_y_one_hot)
    train_accuracies, val_accuracies, test_accuracies, nodes = result
    train_accuracies_list.append(train_accuracies)
    val_accuracies_list.append(val_accuracies)
    test_accuracies_list.append(test_accuracies)
    nodes_list.append(nodes)

# print(f'final-train-d_15-{train_accuracies_list[0][-1]}')
# print(f'final-valid-d_15-{val_accuracies_list[0][-1]}')
# print(f'final-test-d_15-{test_accuracies_list[0][-1]}')
# print(f'final-train-d_25-{train_accuracies_list[1][-1]}')
# print(f'final-valid-d_25-{val_accuracies_list[1][-1]}')
# print(f'final-test-d_25-{test_accuracies_list[1][-1]}')
# print(f'final-train-d_35-{train_accuracies_list[2][-1]}')
# print(f'final-valid-d_35-{val_accuracies_list[2][-1]}')
# print(f'final-test-d_35-{test_accuracies_list[2][-1]}')
# print(f'final-train-d_45-{train_accuracies_list[3][-1]}')
# print(f'final-valid-d_45-{val_accuracies_list[3][-1]}')
# print(f'final-test-d_45-{test_accuracies_list[3][-1]}')

# PLOTS FOR ACCURACIES VS NUMBER OF NODES DURING POST-PRUNING
for i in range(len(max_depths_one_hot)):
    plt.figure()
    plt.plot(nodes_list[i], train_accuracies_list[i], label='train', color='red')
    plt.plot(nodes_list[i], val_accuracies_list[i], label='validation', color='blue')
    plt.plot(nodes_list[i], test_accuracies_list[i], label='test', color='green')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Accuracy vs Nodes: Depth-{max_depths_one_hot[i]}')
    plt.savefig(f'pruning-{max_depths_one_hot[i]}.png')

# CLASS FOR DTREE USING SKLEARN
class DTree_Sklearn:
    def __init__(self, train_x, train_y, max_depth=None, ccp_alpha=0.0):
        self.train_x = train_x
        self.train_y = train_y
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.decision_tree = self.make_decision_tree()

    def make_decision_tree(self):
        decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=self.max_depth, ccp_alpha=self.ccp_alpha)
        decision_tree = decision_tree.fit(self.train_x, self.train_y)
        return decision_tree

    def get_accuracy(self, x, y):
        predictions = self.decision_tree.predict(x)
        accuracy = (np.sum(predictions == y.T)) / len(y)
        return accuracy

# DTREE_SKLEARN MODELS
depths = [15, 25, 35, 45]
ccp_alphas = [0.001, 0.01, 0.1, 0.2]

train_accuracies_depth, validation_accuracies_depth, test_accuracies_depth = [], [], []
train_accuracies_ccp, validation_accuracies_ccp, test_accuracies_ccp = [], [], []

for depth in depths:
    decision_tree_sklearn = DTree_Sklearn(train_x_one_hot, train_y_one_hot, depth, 0.0)
    train_accuracy = decision_tree_sklearn.get_accuracy(train_x_one_hot, train_y_one_hot)
    validation_accuracy = decision_tree_sklearn.get_accuracy(validation_x_one_hot, validation_y_one_hot)
    test_accuracy = decision_tree_sklearn.get_accuracy(test_x_one_hot, test_y_one_hot)
    train_accuracies_depth.append(train_accuracy)
    validation_accuracies_depth.append(validation_accuracy)
    test_accuracies_depth.append(test_accuracy)

for ccp_alpha in ccp_alphas:
    decision_tree_sklearn = DTree_Sklearn(train_x_one_hot, train_y_one_hot, None, ccp_alpha)
    train_accuracy = decision_tree_sklearn.get_accuracy(train_x_one_hot, train_y_one_hot)
    validation_accuracy = decision_tree_sklearn.get_accuracy(validation_x_one_hot, validation_y_one_hot)
    test_accuracy = decision_tree_sklearn.get_accuracy(test_x_one_hot, test_y_one_hot)
    train_accuracies_ccp.append(train_accuracy)
    validation_accuracies_ccp.append(validation_accuracy)
    test_accuracies_ccp.append(test_accuracy)

best_depth = depths[np.argmax(validation_accuracies_depth)]
best_ccp = ccp_alphas[np.argmax(validation_accuracies_ccp)]

# print(f'best depth = {best_depth}')
# print(f'best ccp_alpha = {best_ccp}')

# print(f'train-depth-{train_accuracies_depth}')
# print(f'valid-depth-{validation_accuracies_depth}')
# print(f'test-depth-{test_accuracies_depth}')
# print(f'train-ccp-{train_accuracies_ccp}')
# print(f'valid-ccp-{validation_accuracies_ccp}')
# print(f'test-ccp-{test_accuracies_ccp}')

# BEST MODEL
d_tree_best = DTree_Sklearn(train_x_one_hot, train_y_one_hot, best_depth, best_ccp)
best_train, best_val, best_test = d_tree_best.get_accuracy(train_x_one_hot, train_y_one_hot), d_tree_best.get_accuracy(validation_x_one_hot, validation_y_one_hot), d_tree_best.get_accuracy(test_x_one_hot, test_y_one_hot)

# print(f'train-{best_train}')
# print(f'val-{best_val}')
# print(f'test-{best_test}')

# PLOTS FOR ACCURACIES VS DEPTH
plt.plot(depths, train_accuracies_depth, label='train', color='red')
plt.plot(depths, validation_accuracies_depth, label='validation', color='blue')
plt.plot(depths, test_accuracies_depth, label='test', color='green')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.title(f'Accuracy vs Max Depth')
plt.savefig(f'dt-sk-d.png')

# PLOTS FOR ACCURACIES VS PRUNING PARAMETER
plt.plot(ccp_alphas, train_accuracies_ccp, label='train', color='red')
plt.plot(ccp_alphas, validation_accuracies_ccp, label='validation', color='blue')
plt.plot(ccp_alphas, test_accuracies_ccp, label='test', color='green')
plt.xlabel('ccp_alpha')
plt.ylabel('Accuracy')
plt.legend()
plt.title(f'Accuracy vs ccp_alpha')
plt.savefig(f'dt-sk-ccp.png')

# CLASS FOR RANDOM FOREST
class Random_Forest:
    def __init__(self, train_x, train_y, n_estimators, max_features, min_samples_split):
        
        self.random_forest = None
        self.train_x = train_x
        self.train_y = train_y
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.make_random_forest()

    def make_random_forest(self):
        random_forest = RandomForestClassifier(n_estimators=self.n_estimators, criterion='entropy', max_features=self.max_features, min_samples_split=self.min_samples_split)
        self.random_forest = random_forest

# GRID SEARCH FOR OPTIMAL PARAMETERS
random_forests = {}
n_estimators = [50, 150, 250, 350]
max_features = [0.1, 0.3, 0.5, 0.7, 0.9]
min_samples_splits = [2, 4, 6, 8, 10]

best_oob_score = -1
best_random_forest = None
best_params = []
for n_estimator in n_estimators:
    for max_feature in max_features:
        for min_samples_split in min_samples_splits:
            random_forest = RandomForestClassifier(n_estimators=n_estimator, criterion='entropy', oob_score=True, max_features=max_feature, min_samples_split=min_samples_split)
            random_forest.fit(train_x_one_hot, train_y_one_hot)
            random_forests[(n_estimator, max_feature, min_samples_split)] = random_forest
            
            if (random_forest.oob_score_ > best_oob_score):
                best_oob_score = random_forest.oob_score_
                best_params = [n_estimator, max_feature, min_samples_split]
                best_random_forest = random_forest

best_train_accuracy = np.sum(best_random_forest.predict(train_x_one_hot) == train_y_one_hot) / len(train_y_one_hot)
best_val_accuracy = np.sum(best_random_forest.predict(validation_x_one_hot) == validation_y_one_hot) / len(validation_y_one_hot)
best_test_accuracy = np.sum(best_random_forest.predict(test_x_one_hot) == test_y_one_hot) / len(validation_y_one_hot)

# print(f'best oob score - {best_oob_score}')
# print(f'train - {best_train_accuracy}')
# print(f'valid-{best_val_accuracy}')
# print(f'test-{best_test_accuracy}')

plt.show(block=True)