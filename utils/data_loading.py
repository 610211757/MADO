# coding:utf-8

import os
import numpy as np
import pandas as pd
from pandas import to_datetime
from datetime import datetime

# import matplotlib.pylab as plt

from numpy import *
np.seterr(divide='ignore',invalid='ignore')

import csv
import pickle as pkl
from xlrd import open_workbook
from sklearn.preprocessing import RobustScaler
import h5py




def main_loading_dataset(config, dataset_name, train_or_test="train"):

    if dataset_name=="kdd99":
        if train_or_test == "train":
            if not os.path.exists(os.path.join(config.processed_data_dir, "kdd_train.h5")):
                samples, labels = kdd99(config, seq_length=config.seq_length, seq_step=config.seq_step)
                f = h5py.File(os.path.join(config.processed_data_dir, "kdd_train.h5"),'w')
                f["x"] = samples
                f["y"] = labels
                f.close()
                print("saving file kdd_train.h5 successfully!")
            else:
                f = h5py.File(os.path.join(config.processed_data_dir, "kdd_train.h5"),'r')
                samples = f["x"][:]
                labels = f["y"][:]
                f.close()
                print("loading file kdd_train.h5 successfully!")
            return samples, labels
        else:
            if not os.path.exists(os.path.join(config.processed_data_dir, "kdd_test.h5")):
                samples, labels, index = kdd99_test(config, seq_length=config.seq_length, seq_step=config.seq_step)
                f = h5py.File(os.path.join(config.processed_data_dir, "kdd_test.h5"),'w')
                f["x"] = samples
                f["y"] = labels
                f["index"] = index
                f.close()
                print("saving file kdd_test.h5 successfully!")
            else:
                f = h5py.File(os.path.join(config.processed_data_dir, "kdd_test.h5"),'r')
                samples = f["x"][:]
                labels = f["y"][:]
                index = f["index"][:]
                f.close()
                print("loading file kdd_test.h5 successfully!")
            return samples, labels, index

    elif dataset_name=="swat":
        if train_or_test == "train":
            if not os.path.exists(os.path.join(config.processed_data_dir, "swat_train.h5")):
                samples, labels = swat(config, seq_length=config.seq_swat_length, seq_step=config.seq_swat_step)
                f = h5py.File(os.path.join(config.processed_data_dir, "swat_train.h5"),'w')
                f["x"] = samples
                f["y"] = labels
                f.close()
                print("saving file swat_train.h5 successfully!")
            else:
                f = h5py.File(os.path.join(config.processed_data_dir, "swat_train.h5"),'r')
                samples = f["x"][:]
                labels = f["y"][:]
                f.close()
                print("loading file swat_train.h5 successfully!")
            return samples, labels
        else:
            if not os.path.exists(os.path.join(config.processed_data_dir, "swat_test.h5")):
                samples, labels, index = swat_test(config, seq_length=config.seq_swat_length, seq_step=config.seq_swat_step)
                f = h5py.File(os.path.join(config.processed_data_dir, "swat_test.h5"),'w')
                f["x"] = samples
                f["y"] = labels
                f["index"] = index
                f.close()
                print("saving file swat_test.h5 successfully!")
            else:
                f = h5py.File(os.path.join(config.processed_data_dir, "swat_test.h5"),'r')
                samples = f["x"][:]
                labels = f["y"][:]
                index = f["index"][:]
                f.close()
                print("loading file swat_test.h5 successfully!")
            return samples, labels, index
    
    elif dataset_name=="wadi":
        
        if train_or_test == "train":
            config.seq_length = 10800
            config.seq_step = 100
            if not os.path.exists(os.path.join(config.processed_data_dir, "wadi_train.h5")):
                samples, labels = wadi(config, seq_length=config.seq_length, seq_step=config.seq_step)
                f = h5py.File(os.path.join(config.processed_data_dir, "wadi_train.h5"),'w')
                f["x"] = samples
                f["y"] = labels
                f.close()
                print("saving file wadi_train.h5 successfully!")
            
            else:
                f = h5py.File(os.path.join(config.processed_data_dir, "wadi_train.h5"),'r')
                samples = f["x"][:]
                labels = f["y"][:]
                f.close()
                print("loading file wadi_train.h5 successfully!")
            return samples, labels
        else:
            config.seq_length = 36
            config.seq_step = 12
            if not os.path.exists(os.path.join(config.processed_data_dir, "wadi_test.h5")):
                samples, labels, index = wadi_test(config, seq_length=config.seq_length, seq_step=config.seq_step)
                f = h5py.File(os.path.join(config.processed_data_dir, "wadi_test.h5"),'w')
                f["x"] = samples
                f["y"] = labels
                f["index"] = index
                f.close()
                print("saving file wadi_test.h5 successfully!")
            else:
                f = h5py.File(os.path.join(config.processed_data_dir, "wadi_test.h5"),'r')
                samples = f["x"][:]
                labels = f["y"][:]
                index = f["index"][:]
                f.close()
                print("loading file wadi_test.h5 successfully!")
            return samples, labels, index

    else:
        raise Exception("the dataset doesn't exist here!")


def check_missing_values(data):
    nan_lists = {}

    total_counter = data.shape[0]

    for columname in data.columns:   
        nan_counter = 0  
        record = {}
        record_items = []
        row_i = 0
        for nan in data[columname].isnull():
            if nan:
                nan_counter += 1
                record["nan_counter"] = nan_counter
                record_items.append(row_i)
                record["nan_index"] = record_items
                nan_lists[columname] = record
            row_i += 1
       
    for k, v in nan_lists.items(): 
        print('列名："{}", 共有{}行缺失值/总行数{}'.format(k, v["nan_counter"], total_counter))
        if v["nan_counter"] < 30:
            print(v["nan_index"])
            print('')


def wadi_plot_interpolate_missings(data):

    show_len = 100

    data = data.interpolate(method='nearest')

    flag_show_missing = False

    if flag_show_missing:
    
        a1 = data.iloc[623873-show_len:623879+show_len, 4].values
        a2 = data.iloc[884845-show_len:884851+show_len, 4].values
        a3 = data.iloc[706470-show_len:706476+show_len, 6].values
        a4 = data.iloc[61703-show_len:61713+show_len, 111-4].values
        a5 = data.iloc[384154-show_len:384164+show_len, 111-4].values
        a6 = data.iloc[524280-show_len:524286+show_len, 113-4].values
        a7 = data.iloc[974812-show_len:974818+show_len, 115-4].values

        plt.subplot(7,1,1)
        plt.plot(a1)
        plt.plot(range(show_len,show_len+6), a1[show_len: show_len+6], c='r')
        plt.subplot(7,1,2)
        plt.plot(a2)
        plt.plot(range(show_len,show_len+6), a2[show_len: show_len+6],  c='r')
        plt.subplot(7,1,3)
        plt.plot(a3)
        plt.plot(range(show_len,show_len+6), a3[show_len: show_len+6],  c='r')
        plt.subplot(7,1,4)
        plt.plot(a4)
        plt.plot(range(show_len,show_len+10), a4[show_len: show_len+10],  c='r')
        plt.subplot(7,1,5)
        plt.plot(a5)
        plt.plot(range(show_len,show_len+10), a5[show_len: show_len+10],  c='r')
        plt.subplot(7,1,6)
        plt.plot(a6)
        plt.plot(range(show_len,show_len+6), a6[show_len: show_len+6],  c='r')
        plt.subplot(7,1,7)
        plt.plot(a7)
        plt.plot(range(show_len,show_len+6), a7[show_len: show_len+6],  c='r')
        
        plt.show()

    return data


def wadi_check_attack(my_datetime):

    attack_time_start = []
    attack_time_end = []

    s1=datetime.strptime('10/9/2017 19:25:00','%m/%d/%Y %H:%M:%S')
    e1=datetime.strptime('10/9/2017 19:50:16','%m/%d/%Y %H:%M:%S')
    attack_time_start.append(s1)
    attack_time_end.append(e1)

    s2=datetime.strptime('10/10/2017 10:24:10','%m/%d/%Y %H:%M:%S')
    e2=datetime.strptime('10/10/2017 10:34:10','%m/%d/%Y %H:%M:%S')
    attack_time_start.append(s2)
    attack_time_end.append(e2)

    s3=datetime.strptime('10/10/2017 10:55:00','%m/%d/%Y %H:%M:%S')
    e3=datetime.strptime('10/10/2017 11:24:00','%m/%d/%Y %H:%M:%S')
    attack_time_start.append(s3)
    attack_time_end.append(e3)

    s4=datetime.strptime('10/10/2017 11:07:46','%m/%d/%Y %H:%M:%S')
    e4=datetime.strptime('10/10/2017 11:12:15','%m/%d/%Y %H:%M:%S')
    attack_time_start.append(s4)
    attack_time_end.append(e4)

    s5=datetime.strptime('10/10/2017 11:30:40','%m/%d/%Y %H:%M:%S')
    e5=datetime.strptime('10/10/2017 11:44:50','%m/%d/%Y %H:%M:%S')
    attack_time_start.append(s5)
    attack_time_end.append(e5)

    s6=datetime.strptime('10/10/2017 13:39:30','%m/%d/%Y %H:%M:%S')
    e6=datetime.strptime('10/10/2017 13:50:40','%m/%d/%Y %H:%M:%S')
    attack_time_start.append(s6)
    attack_time_end.append(e6)

    s7=datetime.strptime('10/10/2017 14:48:17','%m/%d/%Y %H:%M:%S')
    e7=datetime.strptime('10/10/2017 14:59:55','%m/%d/%Y %H:%M:%S')
    attack_time_start.append(s7)
    attack_time_end.append(e7)

    s7_2=datetime.strptime('10/10/2017 14:53:44','%m/%d/%Y %H:%M:%S')
    e7_2=datetime.strptime('10/10/2017 15:00:32','%m/%d/%Y %H:%M:%S')
    attack_time_start.append(s7_2)
    attack_time_end.append(e7_2)

    s8=datetime.strptime('10/10/2017 17:40:00','%m/%d/%Y %H:%M:%S')
    e8=datetime.strptime('10/10/2017 17:49:40','%m/%d/%Y %H:%M:%S')
    attack_time_start.append(s8)
    attack_time_end.append(e8)

    s9=datetime.strptime('10/11/2017 10:55:00','%m/%d/%Y %H:%M:%S')
    e9=datetime.strptime('10/11/2017 10:56:27','%m/%d/%Y %H:%M:%S')
    attack_time_start.append(s9)
    attack_time_end.append(e9)

    s10=datetime.strptime('10/11/2017 11:17:54','%m/%d/%Y %H:%M:%S')
    e10=datetime.strptime('10/11/2017 11:31:20','%m/%d/%Y %H:%M:%S')
    attack_time_start.append(s10)
    attack_time_end.append(e10)

    s11=datetime.strptime('10/11/2017 11:36:31','%m/%d/%Y %H:%M:%S')
    e11=datetime.strptime('10/11/2017 11:47:00','%m/%d/%Y %H:%M:%S')
    attack_time_start.append(s11)
    attack_time_end.append(e11)

    s12=datetime.strptime('10/11/2017 11:59:00','%m/%d/%Y %H:%M:%S')
    e12=datetime.strptime('10/11/2017 12:05:00','%m/%d/%Y %H:%M:%S')
    attack_time_start.append(s12)
    attack_time_end.append(e12)

    s13=datetime.strptime('10/11/2017 12:07:30','%m/%d/%Y %H:%M:%S')
    e13=datetime.strptime('10/11/2017 12:10:52','%m/%d/%Y %H:%M:%S')
    attack_time_start.append(s13)
    attack_time_end.append(e13)

    s14=datetime.strptime('10/11/2017 12:16:00','%m/%d/%Y %H:%M:%S')
    e14=datetime.strptime('10/11/2017 12:25:36','%m/%d/%Y %H:%M:%S')
    attack_time_start.append(s14)
    attack_time_end.append(e14)

    s15=datetime.strptime('10/11/2017 15:26:30','%m/%d/%Y %H:%M:%S')
    e15=datetime.strptime('10/11/2017 15:37:00','%m/%d/%Y %H:%M:%S')
    attack_time_start.append(s15)
    attack_time_end.append(e15)

    for i in range(len(attack_time_start)):
        s = to_datetime(attack_time_start[i])
        e = to_datetime(attack_time_end[i])

        if my_datetime >= s and my_datetime <= e:
            return 1

    return 0


def wadi_add_labels(data):

    labels = []

    for i in range(data.shape[0]):
        print("checking {} ...".format(i))
        date = data.iloc[i,1]
        date = to_datetime(date, format="%m/%d/%Y")
        date = date.strftime('%Y/%m/%d %H:%M:%S')[:10]

        time = data.iloc[i,2]
        time = to_datetime(time, format="%H:%M:%S.000 %p")
        time = time.strftime('%Y-%m-%d %H:%M:%S')[-8:]
        
        date_i = "{} {}".format(date, time)
        date_i = to_datetime(date_i)
        label_i = wadi_check_attack(date_i)
        print(label_i)
        labels.append(label_i)

    print(np.shape(labels), sum(labels))

    return labels
 

# --- deal with the WADI data --- #
# seq_length = 10800, seq_step=300
def wadi(config, seq_length, seq_step, num_signals=1):

    file_name = os.path.join(config.data_dir, "wadi//WADI_14days.csv")

    data = pd.read_csv(file_name, header=None, sep=',',skiprows=range(0, 5))
    # print(df.head(9))

    pd.set_option('display.max_columns', None)

    # check_missing_values(data)

    data.drop(data.columns[[50, 51, 86, 87]], axis=1, inplace=True)
    # print("removing cols: 50, 51, 86, 87")
    

    if os.path.exists(os.path.join(config.processed_data_dir, "wadi_labels.h5")):
        f = h5py.File(os.path.join(config.processed_data_dir, "wadi_labels.h5"),'r')
        labels = f["labels"][:]
        print("loading wadi labels for training successfully!")
        f.close()
    else:
        labels = wadi_add_labels(data.iloc[:,0:3])
        f = h5py.File(os.path.join(config.processed_data_dir, "wadi_labels.h5"),'w')
        f["labels"] = labels
        print("saving wadi labels for training successfully!")
        f.close()

    train = np.array(wadi_plot_interpolate_missings(data))

    # print(np.shape(data))   # (1209601, 126)
    # check_missing_values(data)

    train = train[:,3:]

    m, n = train.shape

    max_value = dict()
    min_value = dict()

    # normalization
    for i in range(n):
        # print('i=', i)
        A = max(train[:, i])
        A1 = min(train[:, i])
        # print('A=', A)
        max_value[i] = A
        min_value[i] = A1
   
        if A == A1:
            train[:, i] = train[:, i]
        else:
            train[:, i] = (train[:, i]-A1)/(A-A1)
            
    file = open("log/wadi_maxvalue.pkl",'wb+')
    pkl.dump(max_value, file)
    file.close()

    file = open("log/wadi_minvalue.pkl",'wb+')
    pkl.dump(min_value, file)
    file.close()

    # samples = train[259200:, :]
    samples = train[259200:, :]
    labels = labels[259200:]
    
    # samples = samples[:, [0, 3, 6, 17]]

    # -- apply PCA dimension reduction for multi-variate GAN-AD -- #
    if config.usingPCA:
        from sklearn.decomposition import PCA
        X_n = samples
        # -- the best PC dimension is chosen pc=6 -- #
        n_components = num_signals
        pca = PCA(n_components, svd_solver='full')
        pca.fit(X_n)
        ex_var = pca.explained_variance_ratio_
        pc = pca.components_
        # projected values on the principal component
        T_n = np.matmul(X_n, pc.transpose(1, 0))
        samples = T_n

        file = open("log/wadi_pca.pkl",'wb+')
        pkl.dump(pca, file)
        file.close()
    else:
        num_signals = n

    num_samples = (samples.shape[0] - seq_length) // seq_step

    aa = np.empty([num_samples, int(seq_length/300), num_signals])
    bb = np.empty([num_samples, int(seq_length/300), 1])

    for j in range(num_samples):
        bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length):300], [-1, 1])
        for i in range(num_signals):
            aa[j, :, i] = samples[(j * seq_step):(j * seq_step + seq_length):300, i]

    samples = aa
    labels = bb

    return samples, labels



def wadi_test(config, seq_length, seq_step, num_signals=1):


    file_name = os.path.join(config.data_dir, "wadi//WADI_attackdata.csv")

    data = pd.read_csv(file_name, header=None, sep=',',skiprows=range(0, 1))
    # print(df.head(9))

    pd.set_option('display.max_columns', None)

    # check_missing_values(data)

    data.drop(data.columns[[50, 51, 86, 87]], axis=1, inplace=True)
    # print("removing cols: 50, 51, 86, 87")
    

    if os.path.exists(os.path.join(config.processed_data_dir, "wadi_labels_test.h5")):
        f = h5py.File(os.path.join(config.processed_data_dir, "wadi_labels_test.h5"),'r')
        labels = f["labels"][:]
        print("loading wadi labels for testing successfully!")
        f.close()
    else:
        labels = wadi_add_labels(data.iloc[:,0:3])
        f = h5py.File(os.path.join(config.processed_data_dir, "wadi_labels_test.h5"),'w')
        f["labels"] = labels
        # (172801,) 10133
        print("saving wadi labels for testing successfully!")
        f.close()

    # test = np.array(wadi_plot_interpolate_missings(data))
    data = np.array(data)
    test = data[:,3:]

    m, n = test.shape

    if os.path.exists("log/wadi_maxvalue.pkl"):
        with open("log/wadi_maxvalue.pkl", 'rb') as f:
            max_value = pkl.load(f)
    else:
        raise Exception("there are not object wadi_maxvalue from train set!")  

    if os.path.exists("log/wadi_minvalue.pkl"):
        with open("log/wadi_minvalue.pkl", 'rb') as f:
            min_value = pkl.load(f)
    else:
        raise Exception("there are not object wadi_minvalue from train set!") 

    # normalization
    for i in range(n):
        
        B = max_value[i]
        B1 = min_value[i]
       
        if B == B1:
            test[:, i] = test[:, i]
        else:
            test[:, i] = (test[:, i]-B1)/(B-B1)

    samples = test
    labels = labels
    idx = np.asarray(list(range(0, m)))  # record the idx of each point
    
    # samples = samples[:, [0, 3, 6, 17]]

    # -- apply PCA dimension reduction for multi-variate GAN-AD -- #
    if config.usingPCA:
        from sklearn.decomposition import PCA
        X_n = samples
        # -- the best PC dimension is chosen pc=6 -- #
        n_components = num_signals
        pca = PCA(n_components, svd_solver='full')
        pca.fit(X_n)
        ex_var = pca.explained_variance_ratio_
        pc = pca.components_
        # projected values on the principal component
        T_n = np.matmul(X_n, pc.transpose(1, 0))
        samples = T_n

        file = open("log/wadi_pca.pkl",'wb+')
        pkl.dump(pca, file)
        file.close()
    else:
        num_signals = n

    num_samples_t = (samples.shape[0] - seq_length) // seq_step
    aa = np.empty([num_samples_t, seq_length, num_signals])
    bb = np.empty([num_samples_t, seq_length, 1])
    bbb = np.empty([num_samples_t, seq_length, 1])

    for j in range(num_samples_t):
        bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        bbb[j, :, :] = np.reshape(idx[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        for i in range(num_signals):
            aa[j, :, i] = samples[(j * seq_step):(j * seq_step + seq_length), i]

    samples = aa
    labels = bb
    index = bbb

    return samples, labels, index




# --- deal with the SWaT data --- #
def swat(config, seq_length, seq_step, num_signals=1):
    
    file_name = os.path.join(config.data_dir, "swat/SWaT_Dataset_Normal_v0.xlsx")

    workbook = open_workbook(file_name)
    workbook_sheet_names = workbook.sheet_names()
    sheet = workbook.sheet_by_name(workbook_sheet_names[0])
    # print(sheet.nrows, sheet.ncols)

    row = sheet.nrows
    col = sheet.ncols

    x = []
    y = []

    for i in range(2, row):
        feature = sheet.row_values(i)[1:-1]
        label = sheet.row_values(i)[-1]
        if label.strip().replace(" ","") == "Attack":
            label = 1
        elif label.strip().replace(" ","") == "Normal":
            label = 0
        x.append(feature)
        assert isinstance(label, int)
        y.append(label)

    train = np.array(x)
    labels = np.array(y)

    m, n = train.shape  # m=496800, n=51

    max_value = dict()
    min_value = dict()

    # normalization
    for i in range(n):
        # print('i=', i)
        A = max(train[:, i])
        A1 = min(train[:, i])
        # print('A=', A)
        max_value[i] = A
        min_value[i] = A1
   
        if A == A1:
            train[:, i] = train[:, i]
        else:
            train[:, i] = (train[:, i]-A1)/(A-A1)
            
    file = open("log/swat_maxvalue.pkl",'wb+')
    pkl.dump(max_value, file)
    file.close()

    file = open("log/swat_minvalue.pkl",'wb+')
    pkl.dump(min_value, file)
    file.close()

    samples = train[21600:, :]
    labels = labels[21600:]

    # -- apply PCA dimension reduction for multi-variate GAN-AD -- #
    if config.usingPCA:
        from sklearn.decomposition import PCA
        import DR_discriminator as dr
        X_a = samples
        
        if os.path.exists("log/wadi_pca.pkl"):
            with open("log/wadi_pca.pkl", 'rb') as f:
                pca_a = pkl.load(f)
        else:
            raise Exception("there are not object wadi_pca from train set!")  

        n_components = num_signals
        pc_a = pca_a.components_
        T_a = np.matmul(X_a, pc_a.transpose(1, 0))
        samples = T_a
    else:
        num_signals = n

    num_samples = (samples.shape[0] - seq_length) // seq_step

    aa = np.empty([num_samples, seq_length, num_signals])
    bb = np.empty([num_samples, seq_length, 1])

    for j in range(num_samples):
        bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        for i in range(num_signals):
            aa[j, :, i] = samples[(j * seq_step):(j * seq_step + seq_length), i]

    samples = aa
    labels = bb

    return samples, labels


def swat_test(config, seq_length, seq_step, num_signals=1):

    file_name = os.path.join(config.data_dir, "swat/SWaT_Dataset_Attack_v0.xlsx")

    workbook = open_workbook(file_name)
    workbook_sheet_names = workbook.sheet_names()
    sheet = workbook.sheet_by_name(workbook_sheet_names[0])
    # print(sheet.nrows, sheet.ncols)

    row = sheet.nrows
    col = sheet.ncols

    x = []
    y = []

    for i in range(2, row):
        feature = sheet.row_values(i)[1:-1]
        label = sheet.row_values(i)[-1]
        if label.strip().replace(" ","") == "Attack":
            label = 1
        elif label.strip().replace(" ","") == "Normal":
            label = 0
        x.append(feature)
        assert isinstance(label, int)
        y.append(label)

    test = np.array(x)
    labels = np.array(y)

    m, n = test.shape  # m=449919, n=51

    if os.path.exists("log/swat_maxvalue.pkl"):
        with open("log/swat_maxvalue.pkl", 'rb') as f:
            max_value = pkl.load(f)
    else:
        raise Exception("there are not object swat_maxvalue from train set!")  

    if os.path.exists("log/swat_minvalue.pkl"):
        with open("log/swat_minvalue.pkl", 'rb') as f:
            min_value = pkl.load(f)
    else:
        raise Exception("there are not object swat_minvalue from train set!") 

    # normalization
    for i in range(n):
        
        B = max_value[i]
        B1 = min_value[i]
       
        if B == B1:
            test[:, i] = test[:, i]
        else:
            test[:, i] = (test[:, i]-B1)/(B-B1)
            

    samples = test
    idx = np.asarray(list(range(0, m)))  # record the idx of each point

    # -- apply PCA dimension reduction for multi-variate GAN-AD -- #
    if config.usingPCA:
        from sklearn.decomposition import PCA
        import DR_discriminator as dr
        X_a = samples
        
        if os.path.exists("log/kdd99_pca.pkl"):
            with open("log/kdd99_pca.pkl", 'rb') as f:
                pca_a = pkl.load(f)
        else:
            raise Exception("there are not object kdd99_pca from train set!")  

        n_components = num_signals
        pc_a = pca_a.components_
        T_a = np.matmul(X_a, pc_a.transpose(1, 0))
        samples = T_a

    else:
        num_signals = n

    num_samples_t = (samples.shape[0] - seq_length) // seq_step
    aa = np.empty([num_samples_t, seq_length, num_signals])
    bb = np.empty([num_samples_t, seq_length, 1])
    bbb = np.empty([num_samples_t, seq_length, 1])

    for j in range(num_samples_t):
        bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        bbb[j, :, :] = np.reshape(idx[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        for i in range(num_signals):
            aa[j, :, i] = samples[(j * seq_step):(j * seq_step + seq_length), i]

    samples = aa
    labels = bb
    index = bbb

    return samples, labels, index



def kdd99(config, seq_length, seq_step, num_signals=1):

    train = np.load(os.path.join(config.data_dir, 'kdd99/kdd99_train.npy'))

    print('load kdd99_train from .npy')

    m, n = train.shape  # m=562387, n=34+1

    max_value = dict()
    min_value = dict()

    # normalization
    for i in range(n - 1):
        # print('i=', i)
        A = max(train[:, i])
        A1 = min(train[:, i])
        # print('A=', A)
        max_value[i] = A
        min_value[i] = A1
   
        if A == A1:
            train[:, i] = train[:, i]
        else:
            train[:, i] = (train[:, i]-A1)/(A-A1)
            
    file = open("log/kdd99_maxvalue.pkl",'wb+')
    pkl.dump(max_value, file)
    file.close()

    file = open("log/kdd99_minvalue.pkl",'wb+')
    pkl.dump(min_value, file)
    file.close()

    samples = train[:, 0:n - 1]
    labels = train[:, n - 1]  # the last colummn is label

    # -- apply PCA dimension reduction for multi-variate GAN-AD -- #
    if config.usingPCA:
        from sklearn.decomposition import PCA
        X_n = samples
        # -- the best PC dimension is chosen pc=6 -- #
        n_components = num_signals
        pca = PCA(n_components, svd_solver='full')
        pca.fit(X_n)
        ex_var = pca.explained_variance_ratio_
        pc = pca.components_
        # projected values on the principal component
        T_n = np.matmul(X_n, pc.transpose(1, 0))
        samples = T_n

        file = open("log/kdd99_pca.pkl",'wb+')
        pkl.dump(pca, file)
        file.close()
    else:
        num_signals = n -1


    num_samples = (samples.shape[0] - seq_length) // seq_step

    aa = np.empty([num_samples, seq_length, num_signals])
    bb = np.empty([num_samples, seq_length, 1])

    for j in range(num_samples):
        bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        for i in range(num_signals):
            aa[j, :, i] = samples[(j * seq_step):(j * seq_step + seq_length), i]

    samples = aa
    labels = bb

    return samples, labels


def kdd99_test(config, seq_length, seq_step, num_signals=1):

    test = np.load(os.path.join(config.data_dir, 'kdd99/kdd99_test.npy'))

    print('load kdd99_test from .npy')

    m, n = test.shape  # m1=494021, n1=34+1

    if os.path.exists("log/kdd99_maxvalue.pkl"):
        with open("log/kdd99_maxvalue.pkl", 'rb') as f:
            max_value = pkl.load(f)
    else:
        raise Exception("there are not object kdd99_maxvalue from train set!")  

    if os.path.exists("log/kdd99_minvalue.pkl"):
        with open("log/kdd99_minvalue.pkl", 'rb') as f:
            min_value = pkl.load(f)
    else:
        raise Exception("there are not object kdd99_minvalue from train set!") 

    # normalization
    for i in range(n - 1):
        
        B = max_value[i]
        B1 = min_value[i]
       
        if B == B1:
            test[:, i] = test[:, i]
        else:
            test[:, i] = (test[:, i]-B1)/(B-B1)
            

    samples = test[:, 0:n - 1]
    labels = test[:, n - 1]
    idx = np.asarray(list(range(0, m)))  # record the idx of each point

    # -- apply PCA dimension reduction for multi-variate GAN-AD -- #
    if config.usingPCA:
        from sklearn.decomposition import PCA
        import DR_discriminator as dr
        X_a = samples
        
        if os.path.exists("log/kdd99_pca.pkl"):
            with open("log/kdd99_pca.pkl", 'rb') as f:
                pca_a = pkl.load(f)
        else:
            raise Exception("there are not object kdd99_pca from train set!")  

        n_components = num_signals
        pc_a = pca_a.components_
        T_a = np.matmul(X_a, pc_a.transpose(1, 0))
        samples = T_a

    else:
        num_signals = n -1

    num_samples_t = (samples.shape[0] - seq_length) // seq_step
    aa = np.empty([num_samples_t, seq_length, num_signals])
    bb = np.empty([num_samples_t, seq_length, 1])
    bbb = np.empty([num_samples_t, seq_length, 1])

    for j in range(num_samples_t):
        bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        bbb[j, :, :] = np.reshape(idx[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        for i in range(num_signals):
            aa[j, :, i] = samples[(j * seq_step):(j * seq_step + seq_length), i]

    samples = aa
    labels = bb
    index = bbb

    return samples, labels, index









