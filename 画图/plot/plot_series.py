import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from xlrd import open_workbook

file_name = os.path.join("/home/lee/PycharmProjects/codes 1.4/dataset/swat", "SWaT_Dataset_Attack_v0.xlsx")
print(file_name)
# quit()


# workbook = open_workbook(file_name)
# workbook_sheet_names = workbook.sheet_names()
# sheet = workbook.sheet_by_name(workbook_sheet_names[0])
#
#
# row = sheet.nrows
# col = sheet.ncols
# print(row, col)
#
# x = []
# y = []
# i = 0
# for i in range(2, row-1):
#     i+=1
#     feature = sheet.row_values(i)[1:-1]
#     label = sheet.row_values(i)[-1]
#     if label.strip().replace(" ","") == "Attack":
#         label = 1
#     elif label.strip().replace(" ","") == "Normal":
#         label = 0
#     x.append(feature)
#     assert isinstance(label, int)
#     y.append(label)
#     # if i == 10000:
#     #     break
#     print(i)
#
# with open("log/swat_series_all.pkl", 'wb') as f:
#     pickle.dump([x, y], f)
#
# quit()

with open("log/swat_series_all.pkl", 'rb') as f:
    x, y = pickle.load(f)

x = x[131000:134000]
y = y[131000:134000]

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

attack_section = [1920, 2380]

x0 = list(range(len(x)))
ylabel = [i for i in range(len(y)) if y[i]==1]
attack = list(range(attack_section[0], attack_section[1]))
single_series = train[:,:]
series_y = [single_series[i] for i in range(len(x0)) if y[i]==1]
print(len(series_y))
series_x = [x0[i] for i in range(len(x0)) if y[i]==1]
print(len(series_x))
print(len(x0), len(single_series))

plt.subplot(211)
plt.plot(x0, train[:,:], 'steelblue')  # all time series
for i in range(len(anomaly_section)):
    begin = anomaly_section[i][0]
    end = anomaly_section[i][1]
    print(end-begin)
    # plt.plot(x0[begin:end], single_series[begin:end], "r")
    plt.axvspan(x0[begin], x0[end], facecolor='powderblue', alpha=1)  # 异常值的区域
# plt.plot([attack[i] for i in range(len(attack)) if i%100==0], [train[1757:2696,2][i] for i in range(len(train[1757:2696,2])) if i%100==0],"bo")
for i in [33,35,44]:
    plt.plot(attack, train[attack_section[0]: attack_section[1], i],"r-")  # 受到攻击点

with open("log/swat_score.pkl", 'rb') as f:
    score = pickle.load(f)
score = score[0]

# plt.grid()
# 以下是第二个图

plt.subplot(212)
plt.plot(range(len(score)), score, c='k', lw=0.8)
for i in range(len(anomaly_section)):
    begin = anomaly_section[i][0]
    end = anomaly_section[i][1]
    print(end-begin)
    # plt.plot(x0[begin:end], single_series[begin:end], "r")
    plt.axvspan(x0[begin], x0[end], facecolor='powderblue', alpha=1)



# plt.grid(axis="y")
plt.show()