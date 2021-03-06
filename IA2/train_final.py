import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

def plot(df_lamda_w_zeros, w_zeros_plot_save_path, y_name):
  plt.clf()
  lamda_w_zeros = sns.histplot(data=df_lamda_w_zeros, y=y_name).set_title("weights_zeros_vs_lamda")
  fig_lamda_w_zeros = lamda_w_zeros.get_figure()
  fig_lamda_w_zeros.savefig(w_zeros_plot_save_path)



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
    return df_X.to_numpy(), df_Y.to_numpy()
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


# calculate loss here -- loss function
def sigmoid(z): 
  return 1 / (1 + np.exp(-z))
    
def LR_l2(X, Y, iter_num, lamda, alpha):  
    
  # print(predict_0)
  # print(-np.sum(predict_1 + predict_0) / X.shape[0] + lamda * np.sum((w**2)))

  iter_count = 0
  # w = np.random.rand(X.shape[1])
  w = np.ones(X.shape[1])
  loss_values = []

  while iter_count < iter_num: # and loss_function(X, Y, w, lamda) < epsilon:
    iter_count += 1
    # print(delta_w(X,Y,w))
    # print(w)
    predict_1 = Y * np.log(y_hat(X, w))
    predict_0 = (1 - Y) * np.log(1 - y_hat(X, w))
    w = w + alpha * delta_w(X, Y, w)
    w = w - alpha * lamda * w
    loss_value = -np.sum(predict_1 + predict_0) / X.shape[0] + lamda * np.sum((w**2))
    # print(w)
    # print(loss_function_l2(X, Y, w, lamda))
    loss_values.append(loss_value)

  return w, loss_values

def y_hat(X, w):
  '''
  :param X: np 2d array, N*d, the training data
  :param w: np 1d array, 1*d, the learned parameters
  :return: y_hat, np 1d array, 1*N, the computed y (predicted y)
  '''

  return sigmoid(np.dot(X, w))

def delta_w(X, Y, w):
  '''
  :param X: np 2d array, N*d, training data
  :param Y: np 1d array, 1*N, ground truth
  :param w: np 1d array, 1*d, parameters learned for each iteration
  :return: delta_w, gradient for MSE, used to update w for each iteration
  '''

  return 1 / Y.shape[0] * np.dot(( Y - y_hat(X, w)), X)


# def plot_loss_vs_iter(df_iter_mse, MSE_plot_save_path, learning_rate):
#   '''
#   :param df_iter_mse: df, contains iteration number and MSE value
#   :param MSE_plot_save_path: str, the file name to save the line plot
#   :param learning_rate: the used learning rate
#   :return: none, save lineplot to the file path
#   '''

#   plt.clf()
#   iter_mse = sns.lineplot(data=df_iter_mse, x="iter", y="loss").set_title('learning rate: ' + str(learning_rate))
#   fig_iter_mse = iter_mse.get_figure()
#   fig_iter_mse.show()
#   # fig_iter_mse.savefig(MSE_plot_save_path)


def plot_accuracy_vs_lamda(df_lamda_accuracy, accuracy_plot_save_path, train_or_val):
  '''
  :param df_lamda_accuracy: df, contains accuracy and lamda value (regularization term)
  :param accuracy_plot_save_path: str, the file name to save the line plot
  :param regularization_term: the used regularization term
  :return: none, save lineplot to the file path
  '''

  plt.clf()
  lamda_accuracy = sns.lineplot(data=df_lamda_accuracy, x="regularization_lamda", y="accuracy").set_title(train_or_val)
  fig_lamda_accuracy = lamda_accuracy.get_figure()
  fig_lamda_accuracy.savefig(accuracy_plot_save_path)


def plot_w_zeros_vs_lamda(df_lamda_w_zeros, w_zeros_plot_save_path):
  '''
  :param df_lamda_accuracy: df, contains accuracy and lamda value (regularization term)
  :param accuracy_plot_save_path: str, the file name to save the line plot
  :param regularization_term: the used regularization term
  :return: none, save lineplot to the file path
  '''

  plt.clf()
  lamda_w_zeros = sns.lineplot(data=df_lamda_w_zeros, x="regularization_lamda", y="w_zeros").set_title("weights_zeros_vs_lamda")
  fig_lamda_w_zeros = lamda_w_zeros.get_figure()
  fig_lamda_w_zeros.savefig(w_zeros_plot_save_path)


def feature_engineering(df):
  df['Annual_Premium'] = pd.cut(df['Annual_Premium'],[-100,10000,15000,20000,25000,30000,35000,40000,50000,1000000],labels=list(range(0,9)))
  # df['Age'] = df['Age'].map(pd.qcut(df['Age'].value_counts(), 20, labels=list(range(0,20))))
  # df['Vintage'] = df['Vintage'].map(pd.qcut(df['Vintage'].value_counts(), 8, labels=list(range(0,8))))
  df['Age'] = pd.cut(df['Age'],[-100,28,40,45,50,60,10000],labels=[0,1,2,3,4,5])
  # df['Annual_Premium'] = pd.cut(df['Annual_Premium'],[-100,10000,20000,30000,40000,50000,1000000],labels=[0,1,2,3,4,5])
  df['Vintage'] = pd.cut(df['Vintage'],[-100,50,100,150,200,250,10000],labels=[0,1,2,3,4,5])

  age_dums = pd.get_dummies(df['Age'], prefix="onehot_Age")
  annual_premium_dums = pd.get_dummies(df['Annual_Premium'], prefix="onehot_Annual_Premium")
  vintage_dums = pd.get_dummies(df['Vintage'], prefix="onehot_Vintage")

  df.drop(['Age', 'Annual_Premium', 'Vintage', 'dummy'], axis=1, inplace=True)
  df = pd.concat([df, age_dums, annual_premium_dums, vintage_dums], axis=1)
  

  return df


training_file_path = "./IA2-train.csv"
validation_file_path = "./IA2-dev.csv"
test_file_path = "./IA2-test-small-v2-X.csv"
submission_filename = "./submission.csv"

print("Loading..")
df_train = data_preprocessing(training_file_path)
df_val = data_preprocessing(validation_file_path)
df_test = data_preprocessing(test_file_path)


print("feature_engineering..")
df_train = feature_engineering(df_train)
df_val = feature_engineering(df_val)
df_test = feature_engineering(df_test)

# separate X and Y
X_train, Y_train = separate_X_Y(df_train)
X_val, Y_val = separate_X_Y(df_val)
X_test, _ = separate_X_Y(df_test)



iter_num = 1000
# epsilon = math.pow(10, -6)
# learning rate
alpha = 0.5
# regularization parameter
lamda = 1e-3

print("training..")
# Model training
X_train = np.concatenate((X_train, X_val), axis=0)
Y_train = np.concatenate((Y_train, Y_val), axis=0)
w, loss_values = LR_l2(X_train, Y_train, iter_num, lamda, alpha)
print(loss_values[-1])

print("predict..")
y_train_predicted = y_hat(X_train, w)
y_val_predicted = y_train_predicted

for cutoff in np.arange(0.4, 0.6, 0.05):
  y_train_class = np.array([1 if x >= cutoff else 0 for x in y_train_predicted])
  y_val_class = np.array([1 if x >= cutoff else 0 for x in y_val_predicted])

  train_accuracy = np.count_nonzero(Y_train == y_train_class) / Y_train.shape[0]
  val_accuracy = np.count_nonzero(Y_train == y_val_class) / Y_train.shape[0]

  print("Train: cutoff:%0.2f, accuracy:%0.4f" % (cutoff, train_accuracy))
  print("Val:   cutoff:%0.2f, accuracy:%0.4f" % (cutoff, val_accuracy))
  print()


y_test_predicted = y_hat(X_test, w)
y_test_class = np.array([1 if x >= cutoff else 0 for x in y_test_predicted])
val_pred = pd.DataFrame({'ID': list(range(0,len(y_test_class))),
                        'Response': y_test_class})
val_pred.to_csv(submission_filename, index=False)

