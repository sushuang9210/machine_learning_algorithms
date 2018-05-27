# Random Forest Algorithm on Sonar Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import numpy as np

class RandomForest:
    def __init__(self,data_1,data_2):
        num_data_1 = data_1.shape[0]
        num_data_2 = data_2.shape[0]
        #mark the labels as 1 and -1
        data_1[:,-1] = np.ones((num_data_1))
        data_2[:,-1] = np.zeros((num_data_2))
        #combine the data and shuffle
        self.train_set = np.concatenate((data_1, data_2),axis=0)
        np.random.shuffle(self.train_set)
        #self.train_feature = self.train_set[:,0:-1]
        #self.train_feature = np.concatenate((self.train_set[:,0:-1],-np.ones((2*num_data,1))),axis=1)
        #self.train_label = self.train_set[:,-1]
        #self.w = self.svm_sgd(self.train_feature, self.train_label)
        #print(self.w)

# Calculate accuracy percentage
    def accuracy_metric(self, actual, predicted):
	    correct = 0
	    for i in range(len(actual)):
		    if actual[i] == predicted[i]:
			    correct += 1
	    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self, test_set, *args):
	    output_1 = self.random_forest(self.train_set,test_set,*args)
	    output_2 = np.ones((len(output_1)))-output_1
	    return output_1, output_2

# Split a dataset based on an attribute and an attribute value
    def test_split(self, index, value, dataset):
	    left, right = list(), list()
	    for row in dataset:
		    if row[index] < value:
			    left.append(row)
		    else:
			    right.append(row)
	    return left, right

# Calculate the Gini index for a split dataset
    def gini_index(self, groups, classes):
	# count all samples at split point
	    n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	    gini = 0.0
	    for group in groups:
		    size = float(len(group))
		# avoid divide by zero
		    if size == 0:
			    continue
		    score = 0.0
		# score the group based on the score for each class
		    for class_val in classes:
			    p = [row[-1] for row in group].count(class_val) / size
			    score += p * p
		# weight the group score by its relative size
		    gini += (1.0 - score) * (size / n_instances)
	    return gini

# Select the best split point for a dataset
    def get_split(self, dataset, n_features):
	    class_values = list(set(row[-1] for row in dataset))
	    b_index, b_value, b_score, b_groups = 999, 999, 999, None
	    features = list()
	    while len(features) < n_features:
		    index = randrange(len(dataset[0])-1)
		    if index not in features:
			    features.append(index)
	    for index in features:
		    for row in dataset:
			    groups = self.test_split(index, row[index], dataset)
			    gini = self.gini_index(groups, class_values)
			    if gini < b_score:
				    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	    return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
    def to_terminal(self, group):
	    outcomes = [row[-1] for row in group]
	    return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
    def split(self, node, max_depth, min_size, n_features, depth):
	    left, right = node['groups']
	    del(node['groups'])
	# check for a no split
	    if not left or not right:
		    node['left'] = node['right'] = self.to_terminal(left + right)
		    return
	# check for max depth
	    if depth >= max_depth:
		    node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
		    return
	# process left child
	    if len(left) <= min_size:
		    node['left'] = self.to_terminal(left)
	    else:
		    node['left'] = self.get_split(left, n_features)
		    self.split(node['left'], max_depth, min_size, n_features, depth+1)
	# process right child
	    if len(right) <= min_size:
		    node['right'] = self.to_terminal(right)
	    else:
		    node['right'] = self.get_split(right, n_features)
		    self.split(node['right'], max_depth, min_size, n_features, depth+1)

# Build a decision tree
    def build_tree(self, train, max_depth, min_size, n_features):
	    root = self.get_split(train, n_features)
	    self.split(root, max_depth, min_size, n_features, 1)
	    return root

# Make a prediction with a decision tree
    def predict(self, node, row):
	    if row[node['index']] < node['value']:
		    if isinstance(node['left'], dict):
			    return self.predict(node['left'], row)
		    else:
			    return node['left']
	    else:
		    if isinstance(node['right'], dict):
			    return self.predict(node['right'], row)
		    else:
			    return node['right']

# Create a random subsample from the dataset with replacement
    def subsample(self, dataset, ratio):
	    sample = list()
	    n_sample = round(len(dataset) * ratio)
	    while len(sample) < n_sample:
		    index = randrange(len(dataset))
		    sample.append(dataset[index])
	    return sample

# Make a prediction with a list of bagged trees
    def bagging_predict(self, trees, row):
	    predictions = [self.predict(tree, row) for tree in trees]
	    return max(set(predictions), key=predictions.count)

# Random Forest Algorithm
    def random_forest(self, train, test, max_depth, min_size, sample_size, n_trees, n_features):
	    trees = list()
	    for i in range(n_trees):
		    sample = self.subsample(train, sample_size)
		    tree = self.build_tree(sample, max_depth, min_size, n_features)
		    trees.append(tree)
	    predictions = [self.bagging_predict(trees, row) for row in test]
	    return(predictions)
