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
	feature_name: int
		Index of feature used for splitting on
	split_value: int
		Categorical value for the threshold to split on for the feature
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

  # if np.any(probabilities == 0):
  #   return 0

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

  # print(ori_entropy - (np.dot(freq_array, entropy_array)))

  return ori_entropy - (np.dot(freq_array, entropy_array))


def decision_tree(df, feature_names, depth, max_depth, target_attribute_name = "class"):

  depth = depth + 1
  
  # select the best gain & best feature
  best_gain = 0
  best_feature = feature_names[0]
  class_prediction = 0
  # split_value = 0
  print(depth)

  for feature in feature_names:
    
    gain = info_gain(df, feature, target_name = 'class')
    
    if gain > best_gain:
      best_gain = gain
      best_feature = feature
      unique_values = df[feature].unique()
      if df[df[feature] == unique_values[0]].shape[0] >= df[df[feature] == unique_values[1]].shape[0]:
        class_prediction = df[df[feature] == unique_values[0]]['class'].iloc[0]
        # split_value = unique_values[0]
      else:
        class_prediction = df[df[feature] == unique_values[1]]['class'].iloc[0]
        # split_value = unique_values[1]

  print(best_gain)
  print(best_feature)

  node = Node(class_prediction, best_feature, None, None)
  # stop consition
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

  if (node.left_tree is None) and (node.right_tree is None):
    return node.class_prediction

  if example[node.feature_name] == 0:
    return traverse_tree(node.left_tree, example)
  else:
    return traverse_tree(node.right_tree, example)


def predict(decision_tree_root, df):

  accuracy = 0
  for _, row in df.iterrows():
    predicted_class = traverse_tree(decision_tree_root, row)
    if predicted_class == row['class']:
      accuracy = accuracy + 1
  return accuracy / df.shape[0]


def plot_acc_train_val(df_acc, acc_plot_save_path):
  '''
  :param df_acc: df, contains iter_num as X, and acc of train & val and online & avg as Y
  :param acc_plot_save_path: str, the file name to save the line plot
  :return: none, save lineplot to the file path
  '''

  plt.clf()
  acc = sns.lineplot(data=df_acc)
  acc.set_title('accuracy of train & val for online/avg perceptron')
  acc.set_xlabel("iteration number")
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
  # print(df_train.shape)

  # info_gained = info_gain(df_train, 'habitat=w ')
  # print(info_gained)

  # Model training
  feature_names = df_train.columns.tolist()
  feature_names.remove('class')
  decision_tree_root = decision_tree(df_train, feature_names, -1, dmax, target_attribute_name = "class")
  accuracy_train = predict(decision_tree_root, df_train)
  accuracy_val = predict(decision_tree_root, df_val)
  print(accuracy_train)
  print(accuracy_val)


  # # draw accuracy plot for train & val, online & avg
  # if not os.path.isdir("./plots/"):
  #   os.mkdir("./plots/") 

  # acc_plot_save_path = "./plots/perceptron_acc_train_val.jpg"

  # dict_acc = {
  #   'acc_online_train': acc_ori_list,
  #   'acc_online_val': acc_ori_list_val,
  #   'acc_avg_train': acc_avg_list,
  #   'acc_avg_val': acc_avg_list_val
  # }

  # df_acc = pd.DataFrame(dict_acc)
  # plot_acc_train_val(df_acc, acc_plot_save_path)