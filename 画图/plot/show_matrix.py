import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import pickle



with open("log/swat_series_all.pkl", 'rb') as f:
    x, y = pickle.load(f)

comp_x = np.array(x[21000:21120])
comp_y = np.array(y[21000:21120])

x = x[131000:134000]
y = y[131000:134000]

train = np.array(x)
labels = np.array(y)


m, n = train.shape
attack_section = [1920, 2380]

matrix = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        matrix[i][j] = np.dot(train[2100:2220,i], train[2100:2220,j])
        # matrix[i][j] = np.dot(train[1920:2380, i], train[1920:2380, j])

normal = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        normal[i][j] = np.dot(comp_x[0:120, i], comp_x[0:120, j])
        # normal[i][j] = np.dot(train[0:1000, i], train[0:1000, j])

normal2 = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        normal2[i][j] = np.dot(train[1000:1120, i], train[1000:1120, j])
        # normal[i][j] = np.dot(train[0:1000, i], train[0:1000, j])

# show

# plt.imshow(a) # 显示图片
# plt.imshow(b, cmap='gray')
# plt.axis('off') # 不显示坐标轴

plt.subplot(221); plt.imshow(matrix, cmap='gray')
plt.title("abnormal")
plt.subplot(222); plt.imshow(normal, cmap='gray')
plt.title("normal")
plt.subplot(223); plt.imshow(normal2-normal, cmap='gray')
plt.title("normal-normal")
plt.subplot(224); plt.imshow(matrix-normal, cmap='gray')
plt.title("abnormal-normal")
plt.show()

# save
# 适用于保存任何 matplotlib 画出的图像，相当于一个 screencapture
# plt.savefig('fig_cat.png')
