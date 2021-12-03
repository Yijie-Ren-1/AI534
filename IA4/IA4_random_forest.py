import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random


class Node():
	"""
	Node of decision tree

	Parameters:
	-----------
	class_prediction: int
		Class prediction at this node
	feature_name: str
		feature used for splitting on
	left_tree: Node
		Left subtree
	right_tree: Node
		Right subtree
	"""
	def __init__(self, class_prediction, feature_name, left_tree, right_tree):
		self.class_prediction = class_prediction
		self.feature_name = feature_name
		self.left_tree = left_tree
		self.right_tree = right_tree


def data_preprocessing(csv_file_path):
  '''
  :param csv_file_path: str, training or validation csv file path
  :return: processed df
  '''
  df = pd.read_csv(csv_file_path)

  return df


def entropy(df, column_name):
  '''
  :param df: df after possible splitting, the training data
  :param column_name: str, the target column for entropy calculation
  :return: H(Y), entropy of the target column
  '''

  value_counts = np.bincount(df[column_name])
  probabilities = value_counts / df.shape[0]

  final = 0
  for i in range(len(probabilities)):
    if probabilities[i] != 0:
      final = final + probabilities[i] * math.log(probabilities[i], 2)

  return -final


def info_gain(df, column_name, target_name = 'class'):
  '''
  :param df: df after possible splitting, the training data
  :param column_name: str, the column for entropy calculation
  :param target_name: str, the target column, in this training set, it is always "class"
  :return: H(Y), entropy of the target column
  '''

  ori_entropy = entropy(df, target_name)

  unique_values = df[column_name].unique()
  if len(unique_values) == 1:
    return 0

  value_zero_df = df[df[column_name] == unique_values[0]]
  value_one_df = df[df[column_name] == unique_values[1]]

  freq_array = np.array([value_zero_df.shape[0] / df.shape[0], value_one_df.shape[0] / df.shape[0]])
  entropy_array = np.array([entropy(value_zero_df, target_name), entropy(value_one_df, target_name)])

  return ori_entropy - (np.dot(freq_array, entropy_array))


def decision_tree(df, feature_names, depth, max_depth, target_attribute_name = "class"):
  '''
  :param df: df after possible splitting, the training data
  :param feature_names: list, the columns for best feature selection
  :param depth: int, the current node depth
  :param max_depth: int, varies from 1 - 10, defined in main function
  :param target_attribute_name: str, the target column, in this training set, it is always "class"
  :return: Node, the decidion tree root node
  '''

  depth = depth + 1

  # select the best gain & best feature
  best_gain = 0
  if len(feature_names) > 0:
    best_feature = feature_names[0]
  else:
    best_feature = ''
  class_prediction = 1 if df['class'].sum() / df.shape[0] > 0.5 else 0

  for feature in feature_names:
    
    gain = info_gain(df, feature, target_name = 'class')
    
    if gain > best_gain:
      best_gain = gain
      best_feature = feature

  node = Node(class_prediction, best_feature, None, None)
  # stop condition
  if best_gain == 0 or depth >= max_depth:
    return node

  df_left = df[df[best_feature] == 0].drop(best_feature, axis=1)
  df_right = df[df[best_feature] == 1].drop(best_feature, axis=1)
  feature_names_without_class = df_left.columns.tolist()
  feature_names_without_class.remove('class')
  node.left_tree = decision_tree(df_left, feature_names_without_class, depth, max_depth, target_attribute_name = "class")
  node.right_tree = decision_tree(df_right, feature_names_without_class, depth, max_depth, target_attribute_name = "class")

  return node


def random_forest(df, number_of_trees, number_of_features, max_depth):
  '''
  :param df: df, the training data
  :param number_of_trees: int, how many trees in the random forest
  :param number_of_features: int, the columns for features selected to build the decision tree
  :param max_depth: int, maximum depth of each decision tree in random forest, defined in main function
  :return: a list of Node, which is the random forest
  '''

  random_forest_trees = []
  for tree_num in range(number_of_trees):
    feature_indices = random.sample(range(df.shape[1]-1), number_of_features) + [df.shape[1]-1]
    df_for_single_tree = df[df.columns[feature_indices]]
    feature_names = df_for_single_tree.columns.tolist()
    feature_names.remove('class')
    decision_tree_root = decision_tree(df_for_single_tree, feature_names, -1, max_depth, target_attribute_name = "class")
    random_forest_trees.append(decision_tree_root)

  return random_forest_trees


def traverse_tree(node, example):
  '''
  :param node: Node, the node to traverse
  :param example: df row, the validation example to predict class
  :return: class_prediction, int, the predicted class
  '''

  if (node.left_tree is None) and (node.right_tree is None):
    return node.class_prediction

  if example[node.feature_name] == 0:
    return traverse_tree(node.left_tree, example)
  else:
    return traverse_tree(node.right_tree, example)


def predict(random_forest_trees, df):
  '''
  :param random_forest_trees: list, the root nodes of trained decision trees
  :param df: df, the dataset to get the class prediction
  :return: accuracy, int, the training/validation accuracy
  '''

  accuracy = 0
  for _, row in df.iterrows():

    predicted_classes = np.array([])

    for tree_root in random_forest_trees:
      predicted_class = traverse_tree(tree_root, row)
      predicted_classes = np.append(predicted_classes, predicted_class)
    
    predicted_classes = predicted_classes.astype(int)
    counts = np.bincount(predicted_classes)
    final_predicted_class = np.argmax(counts)
    if final_predicted_class == row['class']:
      accuracy = accuracy + 1

  return accuracy / df.shape[0]


def plot_acc_train(df_acc, acc_plot_save_path, depth):
  '''
  :param df_acc: df, contains dmax as X, and acc of train as Y
  :param acc_plot_save_path: str, the file name to save the line plot
  :param depth: int, the maximum depth of the random forest for figure
  :return: none, save lineplot to the file path
  '''

  plt.clf()
  acc = sns.lineplot(data=df_acc, x = 'x_range', y = 'acc', hue = 'm_value')
  acc.set_title('training accuracy for random forest, dmax = ' + str(depth))
  acc.set_xlabel("number of trees (T)")
  acc.set_ylabel("training accuracy")
  acc_figure = acc.get_figure()
  acc_figure.savefig(acc_plot_save_path)


def plot_acc_val(df_acc, acc_plot_save_path, depth):
  '''
  :param df_acc: df, contains dmax as X, and acc of val as Y
  :param acc_plot_save_path: str, the file name to save the line plot
  :param depth: int, the maximum depth of the random forest for figure
  :return: none, save lineplot to the file path
  '''

  plt.clf()
  acc = sns.lineplot(data=df_acc, x = 'x_range', y = 'acc', hue = 'm_value')
  acc.set_title('validation accuracy for random forest, dmax = ' + str(depth))
  acc.set_xlabel("number of trees (T)")
  acc.set_ylabel("validation accuracy")
  acc_figure = acc.get_figure()
  acc_figure.savefig(acc_plot_save_path)



if __name__ == '__main__':
    
  dmax_list = [1, 2, 5]
  T = [10, 20, 30, 40, 50]
  m = [5, 10, 25, 50]

  training_file_path = "./mushroom-train.csv"
  validation_file_path = "./mushroom-val.csv"

  # read data from csv file
  df_train = data_preprocessing(training_file_path)
  df_val = data_preprocessing(validation_file_path)

  # draw accuracy plot for train & val
  if not os.path.isdir("./plots/"):
    os.mkdir("./plots/") 

  # Model training
  for depth in dmax_list:
    print('depth:')
    print(depth)
    train_acc_plot_save_path = "./plots/random_forest_acc_train_dmax=" + str(depth) + ".jpg"
    val_acc_plot_save_path = "./plots/random_forest_acc_val_dmax=" + str(depth) + ".jpg"

    acc_train_total = []
    acc_val_total = []
    m_value = []
    for feature_num in m:
      print('feature_num (m):')
      print(feature_num)
      acc_train = []
      acc_val = []
      for tree_num in T:
        random_forest_trees = random_forest(df_train, tree_num, feature_num, depth)
        accuracy_train = predict(random_forest_trees, df_train)
        accuracy_val = predict(random_forest_trees, df_val)
        acc_train.append(accuracy_train)
        acc_val.append(accuracy_val)
        m_value.append('m = ' + str(feature_num))
      acc_train_total = acc_train_total + acc_train
      acc_val_total = acc_val_total + acc_val

    dict_acc_train = {
      'acc': acc_train_total,
      'x_range': T*len(m),
      'm_value': m_value
    }
    df_acc_train = pd.DataFrame(dict_acc_train)
    plot_acc_train(df_acc_train, train_acc_plot_save_path, depth)


    dict_acc_val = {
      'acc': acc_val_total,
      'x_range': T*len(m),
      'm_value': m_value
    }
    df_acc_val = pd.DataFrame(dict_acc_val)
    plot_acc_val(df_acc_val, val_acc_plot_save_path, depth)



  

