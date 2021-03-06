import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os


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
  best_feature = feature_names[0]
  class_prediction = 1 if df['class'].sum() / df.shape[0] > 0.5 else 0
  print('depth:')
  print(depth)

  for feature in feature_names:
    
    gain = info_gain(df, feature, target_name = 'class')
    
    if gain > best_gain:
      best_gain = gain
      best_feature = feature

  print('best_feature:')
  print(best_feature)
  print('best_info_gain:')
  print(best_gain)

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


def predict(decision_tree_root, df):
  '''
  :param decision_tree_root: Node, the root node of a trained decision tree
  :param df: df, the dataset to get the class prediction
  :return: accuracy, int, the training/validation accuracy
  '''

  accuracy = 0
  for _, row in df.iterrows():
    predicted_class = traverse_tree(decision_tree_root, row)
    if predicted_class == row['class']:
      accuracy = accuracy + 1
  return accuracy / df.shape[0]


def plot_acc_train_val(df_acc, acc_plot_save_path):
  '''
  :param df_acc: df, contains dmax as X, and acc of train & val as Y
  :param acc_plot_save_path: str, the file name to save the line plot
  :return: none, save lineplot to the file path
  '''

  plt.clf()
  acc = sns.lineplot(data=df_acc, x = 'x_range', y = 'acc', hue = 'train/val acc')
  acc.set_title('accuracy of train & val for decision tree')
  acc.set_xlabel("dmax")
  acc.set_ylabel("training/validation accuracy")
  acc_figure = acc.get_figure()
  acc_figure.savefig(acc_plot_save_path)



if __name__ == '__main__':
    
  dmax = 10

  training_file_path = "./mushroom-train.csv"
  validation_file_path = "./mushroom-val.csv"

  # read data from csv file
  df_train = data_preprocessing(training_file_path)
  df_val = data_preprocessing(validation_file_path)

  # Model training
  acc_train = []
  acc_val = []
  for depth in range(1, dmax+1):
    print("~~~~~~~~~~~~~~~~~~ dmax = " + str(depth) + "~~~~~~~~~~~~~~~~~~~~~~~~~")
    feature_names = df_train.columns.tolist()
    feature_names.remove('class')
    decision_tree_root = decision_tree(df_train, feature_names, -1, depth, target_attribute_name = "class")
    accuracy_train = predict(decision_tree_root, df_train)
    accuracy_val = predict(decision_tree_root, df_val)
    acc_train.append(accuracy_train)
    acc_val.append(accuracy_val)
    print('training accuracy:')
    print(accuracy_train)
    print('validation accuracy:')
    print(accuracy_val)

  # draw accuracy plot for train & val
  if not os.path.isdir("./plots/"):
    os.mkdir("./plots/") 

  acc_plot_save_path = "./plots/decision_tree_acc_train_val.jpg"

  dict_acc = {
    'acc': acc_train + acc_val,
    'x_range': list(range(1, dmax + 1)) + list(range(1, dmax + 1)),
    'train/val acc': ['training_accuracy'] * dmax + ['val_accuracy'] * dmax
  }

  df_acc = pd.DataFrame(dict_acc)
  plot_acc_train_val(df_acc, acc_plot_save_path)