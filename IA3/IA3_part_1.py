import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os


def data_preprocessing(csv_file_path):
  '''
  :param csv_file_path: str, training or validation csv file path
  :return: processed df
  '''
  df = pd.read_csv(csv_file_path)

  return df


def separate_X_Y(df):
  '''
  :param df: processed data
  :return: X: np 2d array, N*d
           Y: np 1d array, 1*N
  '''
  if 'Response' in df:
    df_Y = df['Response']
    df_X = df.drop('Response', axis=1)
    Y_array = df_Y.to_numpy()
    Y_label = np.where(Y_array == 0, -1, 1)
    return df_X.to_numpy(), Y_label
  else:
    return df.to_numpy(), None


def feature_normalization(df_train, df_val):
  '''
  :param df_train: df, training data after preprocessing
  :param df_val: df, validation or test data after preprocessing
  :return: df_train and df_val after normalization of z score
  '''

  # apply z-score to all the columns except "dummy" & "waterfront" & "price"
  columns_applied = ['Annual_Premium', 'Age', 'Vintage']

  # data processing
  # preprocess training data
  df_applied_train = df_train[columns_applied]

  train_mean = mean(df_applied_train)
  train_std = std(df_applied_train)
  df_z_score_train = (df_applied_train - train_mean) / train_std
  df_train.update(df_z_score_train)

  # preprocess validation data
  df_applied_val = df_val[columns_applied]
  df_z_score_val = (df_applied_val - train_mean) / train_std
  df_val.update(df_z_score_val)

  return df_train, df_val


def mean(df):
  return df.mean()

def std(df):
  return df.std()

def z_score(df, mean, std):
  return (df - mean) / std
    

def perceptron(X, Y, iter_num, X_val, Y_val):  
  '''
  :param X: np 2d array, N*d, the training data
  :param Y: np 1d array, 1*N, the training label
  :param iter_num: scalar, the maximum training iteration number
  :param X_val: np 2d array, N*d, the validation data
  :param Y_val: np 1d array, 1*N, the validation label
  :return: w, np 1d array, 1*d, the learned parameters (online perceptron)
           w_avg, np 1d array, 1*d, the learned parameters (average perceptron)
           acc_ori_list, list, length is iter_number, contains the training accuracy of w for each itertion
           acc_avg_list, list, length is iter_number, contains the training accuracy of w_avg for each itertion
           acc_ori_list_val, list, length is iter_number, contains the validation accuracy of w for each itertion
           acc_avg_list_val, list, length is iter_number, contains the validation accuracy of w_avg for each itertion
  '''

  iter_count = 0
  w = np.zeros(X.shape[1])
  w_avg = np.zeros(X.shape[1])
  # example counter
  s = 1 
  acc_ori_list = []
  acc_avg_list = []
  acc_ori_list_val = []
  acc_avg_list_val = []

  while iter_count < iter_num:
    print(iter_count)
    iter_count += 1

    for i in range(len(X)):

      if Y[i] * y_hat(X[i], w) <= 0:
        w = w + Y[i] * X[i]

      w_avg = (s * w_avg + w) / (s + 1)
      s = s + 1 

    y_predicted_ori = [1 if y > 0 else -1 for y in y_hat(X, w)]
    y_predicted_avg = [1 if y > 0 else -1 for y in y_hat(X, w_avg)]
    y_predicted_ori_val = [1 if y > 0 else -1 for y in y_hat(X_val, w)]
    y_predicted_avg_val = [1 if y > 0 else -1 for y in y_hat(X_val, w_avg)]

    acc_ori_value = np.sum(y_predicted_ori == Y) / Y.shape[0]
    acc_avg_value = np.sum(y_predicted_avg == Y) / Y.shape[0]
    acc_ori_value_val = np.sum(y_predicted_ori_val == Y_val) / Y_val.shape[0]
    acc_avg_value_val = np.sum(y_predicted_avg_val == Y_val) / Y_val.shape[0]

    acc_ori_list.append(acc_ori_value)
    acc_avg_list.append(acc_avg_value)
    acc_ori_list_val.append(acc_ori_value_val)
    acc_avg_list_val.append(acc_avg_value_val)

  return w, w_avg, acc_ori_list, acc_avg_list, acc_ori_list_val, acc_avg_list_val


def y_hat(X, w):
  '''
  :param X: np 2d array, N*d, the training data
  :param w: np 1d array, 1*d, the learned parameters
  :return: y_hat, np 1d array, 1*N, the computed y (predicted y)
  '''

  return np.dot(X, w)


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
    
  iter_num = 100

  training_file_path = "./IA2-train.csv"
  validation_file_path = "./IA2-dev.csv"

  df_train = data_preprocessing(training_file_path)
  df_val = data_preprocessing(validation_file_path)

  df_train, df_val = feature_normalization(df_train, df_val)

  # separate X and Y
  X_train, Y_train = separate_X_Y(df_train)
  X_val, Y_val = separate_X_Y(df_val)

  # Model training
  w, w_avg, acc_ori_list, acc_avg_list, acc_ori_list_val, acc_avg_list_val = perceptron(X_train, Y_train, iter_num, X_val, Y_val)

  # draw accuracy plot for train & val, online & avg
  if not os.path.isdir("./plots/"):
    os.mkdir("./plots/") 

  acc_plot_save_path = "./plots/perceptron_acc_train_val.jpg"

  dict_acc = {
    'acc_online_train': acc_ori_list,
    'acc_online_val': acc_ori_list_val,
    'acc_avg_train': acc_avg_list,
    'acc_avg_val': acc_avg_list_val
  }

  df_acc = pd.DataFrame(dict_acc)
  plot_acc_train_val(df_acc, acc_plot_save_path)