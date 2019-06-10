import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from xlrd import open_workbook


with open("log/swat_series_all.pkl", 'rb') as f:
    x, y = pickle.load(f)

type = "Single Stage Multi Point Attacks"

if type == "Single Stage Single Point Attacks":  # attack = 8
    select_section = [14379, 17100]
elif type == "Single Stage Multi Point Attacks": # attack = 37
    select_section = [434520,437329]
elif type == "Multi Stage Single Point Attacks": # attack = 26  # bad
    select_section = [197295,200740]
elif type == "Multi Stage Multi Point Attacks": # attack = 22
    select_section = [131917,134380]

x = x[select_section[0]:select_section[1]]
y = y[select_section[0]:select_section[1]]

train = np.array(x)
labels = np.array(y)

m, n = train.shape

anomaly_section = []
anomaly_section_each = [-1, -1]
for i in range(len(x)-1):
    if y[i] == 0 and y[i+1] == 1:
        anomaly_section_each[0] = i+1
    if y[i] == 1 and y[i+1] == 0:
        anomaly_section_each[1] = i+1
        anomaly_section.append(anomaly_section_each)
        anomaly_section_each = [-1, -1]


x0 = list(range(len(x)))
ylabel = [i for i in range(len(y)) if y[i]==1]

single_series = train[:,:]
series_y = [single_series[i] for i in range(len(x0)) if y[i]==1]
# print(len(series_y))
series_x = [x0[i] for i in range(len(x0)) if y[i]==1]
# print(len(series_x))
# print(len(x0), len(single_series))

plt.figure(figsize=(12,8))
plt.subplot(211)
plt.plot(x0, train[:,:], 'steelblue')  # all time series
for i in range(len(anomaly_section)):
    begin = anomaly_section[i][0]
    end = anomaly_section[i][1]
    # print(i, end-begin, begin, end)
    # plt.plot(x0[begin:end], single_series[begin:end], "r")
    plt.axvspan(x0[begin], x0[end], facecolor='blue', alpha=0.1)  # 异常值的区域


if type == "Single Stage Single Point Attacks":
    attack_section = [1000, 1721]
    attack = list(range(attack_section[0], attack_section[1]))
    for i in [16]:
        plt.plot(attack, train[attack_section[0]: attack_section[1], i],"r-")  # 受到攻击点
elif type == "Multi Stage Multi Point Attacks":
    attack_section = [1000, 1463]
    attack = list(range(attack_section[0], attack_section[1]))
    for i in [33,35,44]:
        plt.plot(attack, train[attack_section[0]: attack_section[1], i],"r-")  # 受到攻击点


# ======================================
# 以下是第二个图
# ======================================

with open("log/swat_score.pkl", 'rb') as f:
    score = pickle.load(f)
score = score[0]
score = score[select_section[0]:select_section[1]]


plt.subplot(212)
plt.plot(range(len(score)), score, c='k', lw=0.8)
for i in range(len(anomaly_section)):
    begin = anomaly_section[i][0]
    end = anomaly_section[i][1]
    # plt.plot(x0[begin:end], single_series[begin:end], "r")
    plt.axvspan(x0[begin], x0[end], facecolor='blue', alpha=0.1)

plt.xlabel("Time steps", fontsize=18)
plt.ylabel("Anomaly score", fontsize=18)

plt.grid(axis="y")
plt.show()



