import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import lightgbm as lgb

def data_preprocessing(csv_file_path):
  '''
    :param csv_file_path: str, training or validation csv file path
    :return: processed df
    '''

  # delete "id" column
  df = pd.read_csv(csv_file_path)
  df_id = df['id']
  df.drop('id', axis=1, inplace=True)

  # split date into month, day, year
  split_items = df["date"].str.split("/", n=2, expand=True)
  df["month"] = split_items[0]
  df["day"] = split_items[1]
  df["year"] = split_items[2]
  df.drop(columns=["date"], inplace=True)

  # add dummy feature column - all 1
  df.insert(0, 'dummy', 1)

  # add new feature age_since_renovated
  df = df.apply(pd.to_numeric)
  for index, row in df.iterrows():
    if row['yr_renovated'] == 0:
      df.loc[index, 'age_since_renovated'] = row['year'] - row['yr_built']
    else:
      df.loc[index, 'age_since_renovated'] = row['year'] - row['yr_renovated']
  df.drop(columns=["yr_renovated"], inplace=True)

  return df, df_id


def separate_X_Y(df):
  '''
    :param df: processed data
    :return: X: np 2d array, N*d
             Y: np 1d array, 1*N
    '''
  if 'price' in df:
    df_price = df['price']
    df_X = df.drop('price', axis=1)
    return df_X.to_numpy(), df_price.to_numpy()
  else:
    return df.to_numpy(), None


def mean(df):
  return df.mean()

def std(df):
  return df.std()

def z_score(df, mean, std):
  return (df - mean) / std


def y_hat(X, w):
  '''
    :param X: np 2d array, N*d, the training data
    :param w: np 1d array, 1*d, the learned parameters
    :return: y_hat, np 1d array, 1*N, the computed y (predicted y)
    '''

  return np.matmul(w, np.transpose(X))


def MSE (X, Y, w):
  '''
    :param X: np 2d array, N*d, training data
    :param Y: np 1d array, 1*N, ground truth
    :param w: np 1d array, 1*d, parameters learned for each iteration
    :return: mse, scalar, mean square error between the predicted y_hat and ground truth y
    '''

  return np.matmul((y_hat(X, w)) - Y, np.transpose((y_hat(X, w)) - Y)) / Y.shape[0]


def delta_w(X, Y, w):
  '''
    :param X: np 2d array, N*d, training data
    :param Y: np 1d array, 1*N, ground truth
    :param w: np 1d array, 1*d, parameters learned for each iteration
    :return: delta_w, gradient for MSE, used to update w for each iteration
    '''

  return 2 / Y.shape[0] * np.matmul((y_hat(X, w)) - Y, X)


def BGD(X, Y, iter_num, learning_r, epsilon):
  '''
    :param X: np 2d array, N*d, training data
    :param Y: np 1d array, 1*N, ground truth
    :return: w, np 1d array, 1*d, finally learned parameters
             MSE_values, list, contains all the MSE calculated during BGD
    '''

  iter_count = 0
  w = np.zeros(shape = (1, X.shape[1]))
  MSE_values = []

  while iter_count < iter_num and MSE(X, Y, w) > epsilon:
    iter_count += 1
    w = w - learning_r * delta_w(X, Y, w)
    MSE_values.append(MSE(X, Y, w)[0,0])

  return w, MSE_values

def plot_MSE_vs_iter(df_iter_mse, MSE_plot_save_path, learning_rate):
  plt.clf()
  iter_mse = sns.lineplot(data=df_iter_mse, x="iter", y="MSE").set_title('learning rate: ' + str(learning_rate))
  fig_iter_mse = iter_mse.get_figure()
  fig_iter_mse.savefig(MSE_plot_save_path)


def mean_squared_error(y_true, y_pred):
  return np.mean((y_true - y_pred)**2)


def feature_normalization(df_train, df_val):
  # apply z-score to all the columns except "dummy" & "waterfront" & "price"
  columns_not_applied = ['waterfront', 'price', 'dummy']
  dummies_columns = [col for col in df_train.columns if 'zipcode' in col]
  dummies_columns += [col for col in df_train.columns if 'has' in col]
  columns_not_applied += dummies_columns

  # data processing
  # preprocess training data
  df_applied_train = df_train[set(df_train.columns) - set(columns_not_applied)]

  train_mean = mean(df_applied_train)
  train_std = std(df_applied_train)
  df_z_score_train = (df_applied_train - train_mean) / train_std
  df_train.update(df_z_score_train)

  # preprocess validation data
  df_applied_val = df_val[set(df_val.columns) - set(columns_not_applied)]
  df_z_score_val = (df_applied_val - train_mean) / train_std
  df_val.update(df_z_score_val)

  return df_train, df_val


if __name__ == '__main__':

  is_kaggle_data = False
  

  if is_kaggle_data is False:
    training_file_path = './IA1_train.csv'
    validation_file_path = './IA1_dev.csv'
    MSE_plot_save_path = './drop_sqrt_living15_learning_r_10_-1.jpg'
    submission_filename = "./submission_results/val_prediction_lgbm.csv"
  else:
    training_file_path = './ai534-ia1/PA1_train1.csv'
    validation_file_path = './ai534-ia1/PA1_test1.csv'
    submission_filename = "./submission_results/submission_lgbm.csv"

  learning_r = math.pow(10, -1)
  iter_num = 5000
  epsilon = 0.001

  df_train, _ = data_preprocessing(training_file_path)
  df_val, df_val_id = data_preprocessing(validation_file_path)

  #   print(df_train[['sqft_above', 'sqft_basement']])
  #   exit(0)

  # feature engineering.
  def feature_enginerring(data):

    # feature 1. onehot zipcode.
    zip_dums = pd.get_dummies(data['zipcode'], prefix="zipcode")
    data = pd.concat([data, zip_dums], axis=1)
    data.drop('zipcode', axis=1, inplace=True)

    # feature 2. total sqft
    data['has_above'] = data['sqft_above'].apply(lambda x: 1 if x > 0 else 0)
    data['has_basement'] = data['sqft_basement'].apply(lambda x: 1
                                                       if x > 0 else 0)

    return data

  # df_train = feature_enginerring(df_train)
  # df_val = feature_enginerring(df_val)

  df_train.to_csv("tmp.csv", index=False)

  # normalization
  # df_train, df_val = feature_normalization(df_train, df_val)

  # separate X and Y
  X_train, Y_train = separate_X_Y(df_train)
  X_val, Y_val = separate_X_Y(df_val)
  
  train_data = lgb.Dataset(X_train, label=Y_train)
  param = {'num_leaves': 8, 'objective': 'mse', "num_iterations": 1000}

  if Y_val is not None:
    validation_data = lgb.Dataset(X_val, label=Y_val)
    model = lgb.train(param, train_data, valid_sets=[validation_data])
  else:
    model = lgb.train(param, train_data)

  y_predicted = model.predict(X_val)
  y_predicted[y_predicted < 0] = 0

  if Y_val is not None:
    # MSE_predicted = MSE(X_val, Y_val, w)
    MSE_predicted = mean_squared_error(Y_val, y_predicted)
    print(MSE_predicted)

  # save submission files.
  val_pred = pd.DataFrame({'id': df_val_id.to_numpy(),
                          'price': y_predicted})
  val_pred.to_csv(submission_filename, index=False)
  # plot_MSE_vs_iter(df_iter_mse, MSE_plot_save_path, learning_r)
