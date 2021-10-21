import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import random
import sys

def np_drop_col(arr, header, names):
    index = np.where(names == header)
    return np.delete(arr, index, 1), np.delete(names, index)

def np_split_col(arr, header, delim, new_headers,names):
    df = pd.DataFrame(arr[1:], columns = names)
    df[new_headers] = df[header].str.split(delim, expand=True)
    df = df.drop(header, axis=1)
    cols = df.columns.tolist()
    cols = cols[-3:] + cols[:-3]
    df = df[cols]
    return df.to_numpy(), cols

def pd_gen_boxplot(arr, header_list):
    arr = arr.astype(float)
    values = dict()
    for header in header_list:
        values[header] = arr[header].unique()
        plt.figure()
        boxplot = arr.boxplot("price", by=header)
        plt.savefig(header)
    return values

def pd_cov_matrix(arr, header_list):
    arr = arr.astype(float)
    sub_arr = arr[header_list]
    return sub_arr.cov()

def pd_scatter(arr, header_list):
    arr = arr.astype(float)
    plt.figure()
    scatterplot = arr.plot.scatter(x=header_list[0], y=header_list[1])
    plt.savefig(str(header_list[0]) + " vs " + str(header_list[1]))

def main(data):
    
    arr = np.loadtxt(data, dtype=str, delimiter=",")
    names = arr[0]
    print("\n####INITIAL DATA####:")    
    print(pd.DataFrame(arr[1:], columns = names))
    
    arr, names = np_drop_col(arr, "id", names)
    print("\n####REMOVED ID####:")    
    print(pd.DataFrame(arr[1:], columns = names))
    
    arr, names = np_split_col(arr, "date", "/", ["month", "day", "year"], names)
    print("\n####SPLIT DATE####:")    
    print(pd.DataFrame(arr, columns = names))

    values = pd_gen_boxplot(pd.DataFrame(arr, columns = names), ['bedrooms', 'bathrooms', 'floors'])
    print("\n####UNIQUE VALUES####:")
    print(values)
    print("\nSEE GENERATED .png FILES OR PDF FOR BOXPLOTS")

    cov_arr = pd_cov_matrix(pd.DataFrame(arr, columns = names), ["sqft_living","sqft_lot","sqft_living15","sqft_lot15"])
    print("\n####CO-VARIANCE MATRIX OF: sqft_living, sqft_lot, sqft_living15, and sqft_lot15####")
    print(cov_arr)

    pd_scatter(pd.DataFrame(arr, columns = names), ["sqft_living", "sqft_living15"])
    pd_scatter(pd.DataFrame(arr, columns = names), ["sqft_lot", "sqft_lot15"])
    print("\nSEE GENERATED .png FILES OR PDF FOR SCATTERPLOTS")
    print("\nANSWERS TO QUESTIONS AND FIGURES IN PDF TITLED 'answers_and_figures.pdf'")



if __name__ == "__main__":
    main(sys.argv[1])
