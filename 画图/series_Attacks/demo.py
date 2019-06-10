import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from xlrd import open_workbook

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

ft_size = 16
shift = -25
end_shift = -19
interval = 16

with open("log/swat_series_all.pkl", 'rb') as f:
    x, y = pickle.load(f)

type = "Multi Stage Multi Point Attacks"

if type == "Single Stage Single Point Attacks":  # attack = 8
    select_section = [14379, 17100]
elif type == "Single Stage Multi Point Attacks": # attack = 37
    select_section = [434520,437329]
elif type == "Multi Stage Single Point Attacks": # attack = 26  # bad
    select_section = [197295,200740]
elif type == "Multi Stage Multi Point Attacks": # attack = 22
    select_section = [131917,134380]

# x = x[select_section[0]:select_section[1]]
# y = y[select_section[0]:select_section[1]]

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

single_series = train[:,0]
series_y = [single_series[i] for i in range(len(x0)) if y[i]==1]
# print(len(series_y))
series_x = [x0[i] for i in range(len(x0)) if y[i]==1]
# print(len(series_x))
# print(len(x0), len(single_series))

plt.figure(figsize=(14,12))
ax1 = plt.subplot(111)
ax1.plot(x0, train[:,0:-1:2], 'steelblue')  # all time series
for i in range(len(anomaly_section)):
    begin = anomaly_section[i][0]
    end = anomaly_section[i][1]
    # print(i, end-begin, begin, end)
    # plt.plot(x0[begin:end], single_series[begin:end], "r")
    ax1.axvspan(x0[begin]+shift, x0[end]+end_shift, facecolor='red', alpha=0.1)  # 异常值的区域

if type == "Single Stage Single Point Attacks":
    attack_section = [1000+shift, 1721+end_shift]
    attack = list(range(attack_section[0], attack_section[1], interval))
    for i in [27]:
        ax1.plot(attack, train[attack_section[0]: attack_section[1]:interval, i],"r^-")  # 受到攻击点
elif type == "Multi Stage Multi Point Attacks":
    attack_section = [1000+shift, 1463+end_shift]
    attack = list(range(attack_section[0], attack_section[1], interval))
    for i in [33,35,44]:
        ax1.plot(attack, train[attack_section[0]: attack_section[1]:interval, i],"r^-")  # 受到攻击点
elif type == "Single Stage Multi Point Attacks":
    attack_section = [2000+shift, 2469+end_shift]
    attack = list(range(attack_section[0], attack_section[1], interval))
    for i in [39,42]:
        ax1.plot(attack, train[attack_section[0]: attack_section[1]:30, i], "r^-")  # 受到攻击点

plt.ylim(0,30)
# plt.xlim(0,2700)

plt.xticks(fontsize=ft_size)
plt.yticks(fontsize=ft_size)
plt.show()



