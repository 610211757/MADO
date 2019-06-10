import matplotlib.pyplot as plt
import numpy as np

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.1, 1.03*height, '%s' % float(height),fontsize=12)


# plt.figure(figsize=(10,4))
plt.figure(figsize=(12,4))

pre = [46.48, 41.37, 38.51, 49.43]
rec = [15.85,28.21,32.83,33.71]
f1 = [23.64,33.55,35.45, 40.08]

# plt.title('Ablation anaylsis on WADI dataset.')
# plt.title('WADI')
name_list = ['Ours w/o GCN or CNN',  'Ours w/o GCN', 'Ours w/o CNN', 'Ours w/ all']


x = list(range(len(pre)))
total_width, n = 0.7, 3
width = total_width / n

a=plt.bar(x, pre, width=width, label='Precision', fc="#718dbf")
for i in range(len(x)):
    x[i] = x[i] + width
b=plt.bar(x, rec, width=width, label='Recall', tick_label=name_list, fc="#c9d9d3")
for i in range(len(x)):
    x[i] = x[i] + width
c=plt.bar(x, f1, width=width, label='F1', fc="salmon")

autolabel(a)
autolabel(b)
autolabel(c)

plt.legend(fontsize=12)
plt.xticks(fontsize=14)
plt.yticks([0,10,20,30,40,50,60,70,80],('0','10%','20%','30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'),fontsize=12)
plt.ylim(0,70)

plt.show()