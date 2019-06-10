# coding:utf-8


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# import matplotlib.pylab as plt
from sklearn import preprocessing

# from tsfresh import extract_features


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def compute_corr_matrix(data):

    # batch_size * seq_len * fea_dims

    num_samples, seq_len, fea_dims = np.shape(data)

    adj = np.zeros([num_samples, fea_dims, fea_dims])

    for i in range(num_samples):

        print("processing sample {}/{} ...".format(i, num_samples))

        for j in range(fea_dims):
            for k in range(fea_dims):
                adj[i, j, k] = np.dot(data[i, :, j], data[i, :, k])
                adj[i, j, k] /= seq_len

        adj_ = adj[i, :, :]
        adj_ = adj_ + np.multiply(adj_.T, adj_.T > adj_) - np.multiply(adj_, adj_.T > adj_)
        adj_ = normalize(adj_ + np.eye(adj_.shape[0]))
        adj[i, :, :] = adj_

    return adj



"""
def compute_ts_all_stat_features(data):

    num_samples, seq_len, fea_dims = np.shape(data)

    bacth_size_here = 512

    i = 0
    
    while (i < num_samples):  

        print("processing sample {}/{} ...".format(i, num_samples))

        if (i+bacth_size_here) < (num_samples-1):
            data_temp = data[i:(i+bacth_size_here)]
        else:
            data_temp = data[i:num_samples]

        data_temp = compute_ts_stat_features(data_temp)
        print(np.shape(data_temp))

        if i == 0:
            res = data_temp
        else:
            res = np.concatenate((res, data_temp), axis=0)

        i += bacth_size_here
    print(np.shape(res))
    return res

# https://github.com/ThomasCai/tsfresh-feature-translation

# computing stat features for each batch
def compute_ts_stat_features(data):

    # data = data.cpu()
    batch_size, seq_len, fea_dims = np.shape(data)
    data = np.transpose(data, (1, 2, 0))

    # print(np.shape(data))

    res = np.zeros((seq_len*fea_dims, batch_size))
    id_time = np.zeros((seq_len*fea_dims, 2))

    count = 0
    for i in range(fea_dims):
        for j in range(seq_len):
            res[count] = data[j, i][:]
            id_time[count, 0] = i
            id_time[count, 1] = j
            count += 1

    timeseries = np.hstack((id_time, res))

    # print(np.shape(timeseries))

    timeseries = pd.DataFrame(timeseries)

    extracted_features = extract_features(timeseries, column_id=0, column_sort=1)

    extracted_features = np.array(extracted_features)
    # print(np.shape(extracted_features))

    extracted_features = np.split(extracted_features, batch_size, axis=1)

    # print(np.shape(extracted_features))

    extracted_features = np.transpose(extracted_features, (0, 2, 1))

    return extracted_features



def compute_per_ts_stat_features(data):

    # data = data.cpu()
    seq_len, fea_dims = np.shape(data)

    # print(np.shape(data))

    res = np.zeros((seq_len*fea_dims, 1))
    id_time = np.zeros((seq_len*fea_dims, 2))

    count = 0
    for i in range(fea_dims):
        for j in range(seq_len):
            res[count] = data[j, i]
            id_time[count, 0] = i
            id_time[count, 1] = j
            count += 1

    timeseries = np.hstack((id_time, res))

    # print(np.shape(timeseries))

    timeseries = pd.DataFrame(timeseries)

    extracted_features = extract_features(timeseries, column_id=0, column_sort=1)

    extracted_features = np.array(extracted_features)

    # print(np.shape(extracted_features))

    extracted_features = np.transpose(extracted_features, (0, 1))

    return extracted_features

"""








