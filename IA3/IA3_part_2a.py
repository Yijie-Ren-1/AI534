import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
import timeit


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
    

def kernel_func(x_1, x_2, p):
  '''
  :param x_1: np 2d array, N*d, the training data
  :param x_2: np 2d array, N*d, the training data
  :param p: scalar, degree of the matrix
  :return: kernalized training data (Gram Matrix)
  '''
  return np.power(np.matmul(x_1, x_2.T), p)


def kernalized_perceptron(X, Y, iter_num, X_val, Y_val, p):  
  '''
  :param X: np 2d array, N*d, the training data
  :param Y: np 1d array, 1*N, the training label
  :param iter_num: scalar, the maximum training iteration number
  :param X_val: np 2d array, N*d, the validation data
  :param Y_val: np 1d array, 1*N, the validation label
  :param p: scalar, the parameter of kernal function
  :return: alpha, np 1d array, 1*N, the learned parameters 
           acc_train_list, list, length is iter_number, contains the training accuracy of each itertion
           acc_val_list, list, length is iter_number, contains the validation accuracy of each itertion
  '''

  iter_count = 0
  alpha = np.zeros(X.shape[0])
  kernalized_x_train = kernel_func(X, X, p)
  kernalized_x_val = kernel_func(X, X_val, p)
 
  acc_train_list = []
  acc_val_list = []

  while iter_count < iter_num:
    print(iter_count)
    iter_count += 1

    for i in range(len(X)):
      u = np.dot(np.multiply(alpha, kernalized_x_train[i]), Y)
      if u * Y[i] <= 0:
        alpha[i] = alpha[i] + 1

    y_predicted_train = y_hat(kernalized_x_train, alpha, Y)
    y_predicted_val = y_hat(kernalized_x_val, alpha, Y)

    acc_train_value = np.sum(y_predicted_train == Y) / Y.shape[0]
    acc_val_value = np.sum(y_predicted_val == Y_val) / Y_val.shape[0]

    acc_train_list.append(acc_train_value)
    acc_val_list.append(acc_val_value)

  return alpha, acc_train_list, acc_val_list


def y_hat(kernalized_X, alpha, Y):
  '''
  :param kernalized_X: np 2d array, N*d, the kernalized trainig/validation data
  :param alpha: np 1d array, 1*N, the learned parameters
  :param Y: np 1d array, 1*N, the trainig label
  :return: y_hat, np 1d array, 1*N, the computed y (predicted y)
  '''

  return np.sign(np.matmul(np.multiply(alpha, Y), kernalized_X))


def kernalized_perceptron_runtime(X, iter_num, Y):
  '''
  :param X: np 2d array, N*d, the training data
  :param iter_num: scalar, the maximum training iteration number
  :param Y: np 1d array, 1*N, the training label
  :return: runtime, scalar, the total time used for training the kernelized model
  '''

  iter_count = 0
  alpha = np.zeros(X.shape[0])
  kernalized_x_train = kernel_func(X, X, 1)

  start = timeit.default_timer()

  while iter_count < iter_num:
    print(iter_count)
    iter_count += 1

    for i in range(len(X)):
      u = np.dot(np.multiply(alpha, kernalized_x_train[i]), Y)
      if u * Y[i] <= 0:
        alpha[i] = alpha[i] + 1
    
  stop = timeit.default_timer()

  return stop - start


def plot_acc_train_val(df_acc, acc_plot_save_path, p):
  '''
  :param df_acc: df, contains iter_num as X, and acc of train & val and online & avg as Y
  :param acc_plot_save_path: str, the file name to save the line plot
  :param p: scalar, the parameter of kernal function
  :return: none, save lineplot to the file path
  '''

  plt.clf()
  acc = sns.lineplot(data=df_acc)
  acc.set_title('accuracy of train & val for kernalized perceptron, p = ' + str(p))
  acc.set_xlabel("iteration number")
  acc.set_ylabel("training/validation accuracy")
  acc_figure = acc.get_figure()
  acc_figure.savefig(acc_plot_save_path)

def plot_runtime_vs_training_size(df_runtime, runtime_plot_save_path):
  '''
  :param df_runtime: df, contains traning dataset size as X, and runtime as Y
  :param acc_plot_save_path: str, the file name to save the scattered plot
  :return: none, save lineplot to the file path
  '''
  
  plt.clf()
  runtime = sns.scatterplot(data=df_runtime, x="training size", y="runtime(second)").set_title('runtime for different training dataset size, p = 1')
  runtime_figure = runtime.get_figure()
  runtime_figure.savefig(runtime_plot_save_path)



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


  if not os.path.isdir("./plots/"):
    os.mkdir("./plots/") 

  # draw accuracy plot for train & val
  for i in range(1, 6):
    print('p =', i)
    # Model training
    alpha, acc_train_list, acc_val_list = kernalized_perceptron(X_train, Y_train, iter_num, X_val, Y_val, i)

    acc_plot_save_path = "./plots/kernelized_perceptron_acc_train_val_p_" + str(i) +".jpg"

    dict_acc = {
      'acc_train_p=' + str(i): acc_train_list,
      'acc_val_p=' + str(i): acc_val_list
    }

    df_acc = pd.DataFrame(dict_acc)
    plot_acc_train_val(df_acc, acc_plot_save_path, i)

  # # draw runtime plot for different training size
  # runtime_plot_save_path = "./plots/runtime_vs_training_size.jpg"
  # X_train_val = np.concatenate((X_train, X_val), axis=0)
  # Y_train_val = np.concatenate((Y_train, Y_val), axis=0)
  # runtime_list = []

  # for power in range(1,5):
  #   runtime = kernalized_perceptron_runtime(X_train_val[:10 ** power,:], iter_num, Y_train_val[:10 ** power])
  #   runtime_list.append(runtime)
  
  # dict_runtime = {
  #   'training size': [10 ** i for i in range(1,5)],
  #   'runtime(second)': runtime_list
  # }

  # df_runtime = pd.DataFrame(dict_runtime)
  # plot_runtime_vs_training_size(df_runtime, runtime_plot_save_path)