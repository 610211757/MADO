import matplotlib.pyplot as plt
import numpy as np

pre = np.array([94.91, 99.40, 97.76, 99.06]) -90
rec = np.array([99.27, 92.82, 97.38, 98.68]) -90
f1 =  np.array([97.04, 96.00, 97.57, 98.87]) -90


# plt.title('WADI')
name_list = ['w/o GCN', 'w/o CNN', 'w/o GCN and CNN', 'Ours']

# plt.title('KDD Cup99')

x = list(range(len(pre)))
total_width, n = 0.5, 3
width = total_width / n

plt.yticks([0, 5, 10],('90','95', '100'))

plt.bar(x, pre, width=width, label='Precious', fc="#718dbf")
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, rec, width=width, label='Recall', tick_label=name_list, fc="#c9d9d3")
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, f1, width=width, label='F1', fc="salmon")
# plt.legend()
plt.legend(loc = 4)
plt.grid(axis="y")
plt.show()


