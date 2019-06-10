

import matplotlib.pylab as plt
import numpy as np


dataset_name = 'kdd99'  # 'swat', 'wadi'


results = np.loadtxt("{}.txt".format(dataset_name))

plt.figure()
plt.plot(results[:,0], color='green')
plt.plot(results[:,1], color='#1F77B4')
plt.plot(results[:,2], color='#FF851B')
max_index = np.argmax(results[:,2])
max_value = np.max(results[:,2])

plt.annotate(s="({}, {}%)".format(max_index, round(100*max_value,2)), xy=(max_index, max_value), xytext=(max_index-15, max_value-0.3), fontsize=14, arrowprops=dict(facecolor="r",arrowstyle="->"))
plt.plot(max_index, max_value, 'o', color="r")
plt.xlim([0,100])
plt.ylim([0,1.1])
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],('0','10%','20%','30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'))
plt.legend(['Precision', 'Recall', 'F1-value'], fontsize=14)
plt.grid(linestyle=":", color="r")

dataset_name = 'swat'  # 'swat', 'wadi'


results = np.loadtxt("{}.txt".format(dataset_name))

plt.figure()
plt.plot(results[:,0], color='green')
plt.plot(results[:,1], color='#1F77B4')
plt.plot(results[:,2], color='#FF851B')
max_index = np.argmax(results[:,2])
max_value = np.max(results[:,2])

plt.annotate(s="({}, {}%)".format(max_index, round(100*max_value,2)), xy=(max_index, max_value), xytext=(max_index-29, max_value+0.1), fontsize=14, arrowprops=dict(facecolor="r",arrowstyle="->"))
plt.plot(max_index, max_value, 'o', color="r")
plt.xlim([0,100])
plt.ylim([0,1])
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],('0','10%','20%','30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'))
plt.legend(['Precision', 'Recall', 'F1-value'], fontsize=14)
plt.grid(linestyle=":", color="r")

dataset_name = 'wadi'  # 'swat', 'wadi'


results = np.loadtxt("{}.txt".format(dataset_name))

plt.figure()
plt.plot(results[:,0], color='green')
plt.plot(results[:,1], color='#1F77B4')
plt.plot(results[:,2], color='#FF851B')
max_index = np.argmax(results[:,2])
max_value = np.max(results[:,2])

plt.annotate(s="({}, {}%)".format(max_index, round(100*max_value,2)), xy=(max_index, max_value), xytext=(max_index-24, max_value+0.2),fontsize=14, arrowprops=dict(facecolor="r",arrowstyle="->"))
plt.plot(max_index, max_value, 'o', color="r")
plt.xlim([0,100])
plt.ylim([0,1])
plt.legend(['Precision', 'Recall', 'F1-value'], fontsize=14)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],('0','10%','20%','30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'))
plt.grid(linestyle=":", color="r")
plt.show()

