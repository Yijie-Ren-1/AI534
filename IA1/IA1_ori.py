import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def data_preprocessing(csv_file_path):
    '''
    :param csv_file_path: str, training or validation csv file path
    :return: processed df
    '''

    # delete "id" column
    df = pd.read_csv(csv_file_path)
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

    return df


def separate_X_Y(df):
    '''
    :param df: processed data
    :return: X: np 2d array, N*d
             Y: np 1d array, 1*N
    '''

    df_price = df['price']
    df_X = df.drop('price', axis=1)

    return df_X.to_numpy(), df_price.to_numpy()


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


if __name__ == '__main__':

    training_file_path = '/Users/yijieren/Downloads/desktop/oregon/courses/AI_534_machine_learning_Fall_21/IA1/IA1_train.csv'
    validation_file_path = '/Users/yijieren/Downloads/desktop/oregon/courses/AI_534_machine_learning_Fall_21/IA1/IA1_dev.csv'
    MSE_plot_save_path = './drop_sqrt_living15_learning_r_10_-1.jpg'

    learning_r = math.pow(10, -1)
    iter_num = 5000
    epsilon = 0.001

    df_train = data_preprocessing(training_file_path)
    df_val = data_preprocessing(validation_file_path)

    # apply z-score to all the columns except "dummy" & "waterfront" & "price"
    columns_not_applied = ['waterfront', 'price', 'dummy']

    # preprocess training data
    df_applied_train = df_train[set(df_train.columns) - set(columns_not_applied)]

    df_train['sqft_above'] = df_train['sqft_above'] + df_train['sqft_basement']
    df_train.drop('sqft_basement', axis=1, inplace=True)
    df_train.grade = np.where(df_train.grade < 4, 0, df_train.grade)
    df_train.grade = np.where((df_train.grade < 11) & (df_train.grade > 4), 1, df_train.grade)
    df_train.grade = np.where(df_train.grade > 10, 2, df_train.grade)


    train_mean = mean(df_applied_train)
    train_std = std(df_applied_train)
    df_z_score_train = (df_applied_train - train_mean) / train_std
    df_train.update(df_z_score_train)
    df_train.drop('sqft_living15', axis=1, inplace=True)


    # preprocess validation data
    df_applied_val = df_val[set(df_val.columns) - set(columns_not_applied)]

    df_val['sqft_above'] = df_val['sqft_above'] + df_val['sqft_basement']
    df_val.drop('sqft_basement', axis=1, inplace=True)
    df_val.grade = np.where(df_val.grade < 4, -1, df_val.grade)
    df_val.grade = np.where((df_val.grade < 11) & (df_val.grade > 4), 0, df_val.grade)
    df_val.grade = np.where(df_val.grade > 10, 1, df_val.grade)

    df_z_score_val = (df_val - train_mean) / train_std
    df_val.update(df_z_score_val)
    df_val.drop('sqft_living15', axis=1, inplace=True)


    # separate X and Y
    X_train, Y_train = separate_X_Y(df_train)
    X_val, Y_val = separate_X_Y(df_val)

    w, MSE_values = BGD(X_train, Y_train, iter_num, learning_r, epsilon)
    dict = {
        'iter': list(range(len(MSE_values))),
        'MSE': MSE_values
    }
    df_iter_mse = pd.DataFrame(dict)

    y_predicted = y_hat(X_val, w)
    MSE_predicted = MSE(X_val, Y_val, w)
    print(w)
    print(MSE_predicted)

    # plot_MSE_vs_iter(df_iter_mse, MSE_plot_save_path, learning_r)



