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

  probabilities = [df[df[column_name] == 0]['distribution'].sum(), df[df[column_name] == 1]['distribution'].sum()]

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
  
  value_zero_df = df[df[column_name] == unique_values[0]]

  if len(unique_values) == 1:
    return 0 # ori_entropy - value_zero_df['distribution'].sum() * entropy(value_zero_df, target_name)

  
  value_one_df = df[df[column_name] == unique_values[1]]

  freq_array = np.array([value_zero_df['distribution'].sum(), value_one_df['distribution'].sum()])
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
  best_feature = feature_names[0]
  df_copy = df.copy()
  df_copy.loc[df_copy['class'] == 0, 'class'] = -1
  values = df_copy['class'] * df_copy['distribution']
  class_prediction = 1 if values.sum() > 0 else 0


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
  feature_names_without_class.remove('distribution')
  node.left_tree = decision_tree(df_left, feature_names_without_class, depth, max_depth, target_attribute_name = "class")
  node.right_tree = decision_tree(df_right, feature_names_without_class, depth, max_depth, target_attribute_name = "class")

  return node


def normalize_D(D):
  '''
  :param D: list, the weights used for adaboost update
  :return: list, normalized list D
  '''
  D_array = np.array(D)
  D_sum = np.sum(D_array)
  D_final = D_array / D_sum
  return D_final.tolist()


def adaboost(df, number_of_trees, max_depth):
  '''
  :param df: df, the training data
  :param number_of_trees: int, L in the course slides, how many ensemble trees in the output
  :param max_depth: int, maximum depth of each decision tree in the output, defined in main function
  :return: a list of Node, which are the ensemble trees (roots)
           a list of weights alpha, weights for each ensemble tree
  '''

  D = np.full((df.shape[0]), 1/df.shape[0])
  alpha = []
  ensemble_trees = []

  for tree_num in range(number_of_trees):
    df['distribution'] = D
    feature_names = df.columns.tolist()
    feature_names.remove('class')
    feature_names.remove('distribution')
    decision_tree_root = decision_tree(df, feature_names, -1, max_depth, target_attribute_name = "class")
    error, error_indices = decision_tree_predict(decision_tree_root, df)
    alpha_tmp = 0.5 * math.log((1 - error) / error)
    alpha.append(alpha_tmp)
    ensemble_trees.append(decision_tree_root)
    for i in range(len(D)):
      if i in error_indices:
        D[i] = D[i] * (math.e ** alpha_tmp)
      else:
        D[i] = D[i] * (math.e ** (-alpha_tmp))
    D = normalize_D(D)

  return ensemble_trees, alpha


def traverse_tree(node, example):
  '''
  :param node: Node, the node to traverse
  :param example: df row, the validation example to predict class
  :return: error, float, total error weights
           error_indices, set, contains the indices of wrongly-predicted examples
  '''

  if (node.left_tree is None) and (node.right_tree is None):
    return node.class_prediction

  if example[node.feature_name] == 0:
    return traverse_tree(node.left_tree, example)
  else:
    return traverse_tree(node.right_tree, example)


def decision_tree_predict(decision_tree_root, df):
  '''
  :param decision_tree_root: Node, the root node of a trained decision tree
  :param df: df, the dataset to get the class prediction
  :return: error, float, the total error weights
  '''

  error = 0
  error_indices = set()
  for index, row in df.iterrows():
    predicted_class = traverse_tree(decision_tree_root, row)
    if predicted_class != row['class']:
      error = error + df['distribution'][index]
      error_indices.add(index)
  return error, error_indices


def adaboost_predict(ensemble_trees, df, alpha):
  '''
  :param ensemble_trees: list, the root nodes of trained ensemble decision trees
  :param df: df, the dataset to get the class prediction
  :param alpha: list, contains weights for each ensemble tree
  :return: accuracy, int, the training/validation accuracy
  '''

  accuracy = 0
  for _, row in df.iterrows():

    predicted_classes = np.array([])

    for tree_root in ensemble_trees:
      predicted_class = traverse_tree(tree_root, row)
      if predicted_class == 0:
         predicted_class = -1
      predicted_classes = np.append(predicted_classes, predicted_class)
    
    final_predicted_class = np.dot(predicted_classes, np.array(alpha))
    if final_predicted_class <= 0:
      final_predicted_class = 0
    else:
      final_predicted_class = 1
    if final_predicted_class == row['class']:
      accuracy = accuracy + 1

  return accuracy / df.shape[0]


def plot_acc_train_val(df_acc, acc_plot_save_path, depth):
  '''
  :param df_acc: df, contains dmax as X, and acc of train as Y
  :param acc_plot_save_path: str, the file name to save the line plot
  :param depth: int, the maximum depth of the adaboost for figure
  :return: none, save lineplot to the file path
  '''

  plt.clf()
  acc = sns.lineplot(data=df_acc, x = 'x_range', y = 'acc', hue = 'train/val acc')
  acc.set_title('training/validation accuracy for adaboost, dmax = ' + str(depth))
  acc.set_xlabel("number of trees (T)")
  acc.set_ylabel("training/validation accuracy")
  acc_figure = acc.get_figure()
  acc_figure.savefig(acc_plot_save_path)



if __name__ == '__main__':
    
  dmax_list = [1, 2, 5]
  T = [10, 20, 30, 40, 50]

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
    acc_plot_save_path = "./plots/adaboost_acc_train_val_dmax=" + str(depth) + ".jpg"

    acc_train = []
    acc_val = []
    for tree_num in T:
      ensemble_trees, alpha = adaboost(df_train, tree_num, depth)
      accuracy_train = adaboost_predict(ensemble_trees, df_train, alpha)
      accuracy_val = adaboost_predict(ensemble_trees, df_val, alpha)
      acc_train.append(accuracy_train)
      acc_val.append(accuracy_val)

    dict_acc = {
      'acc': acc_train + acc_val,
      'x_range': T*2,
      'train/val acc': ['training_accuracy'] * len(T) + ['val_accuracy'] * len(T)
    }
    df_acc = pd.DataFrame(dict_acc)
    plot_acc_train_val(df_acc, acc_plot_save_path, depth)




  

