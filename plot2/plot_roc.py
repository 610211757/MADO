import pickle
from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import  numpy as np



# tsne_label = np.squeeze(tsne_label)
#
# print(tsne_sample.shape, tsne_label.shape)
# tsne = TSNE()
# # pca = PCA(n_components=2)
# X_embedded = tsne.fit_transform(tsne_sample)
# print(type(X_embedded), len(X_embedded), X_embedded.shape)
# colorsss = ['b', 'g']
#
# print(len([1 for x in tsne_label if x == 0]))
# print(len([1 for x in tsne_label if x == 1]))
#
# print(len(tsne_label == 0))
#
# d = X_embedded[tsne_label == 0]
# plt.scatter(d[:, 0], d[:, 1], color=colorsss[0], marker='.', label="normal")
#
# d = X_embedded[tsne_label == 1]
# plt.scatter(d[:, 0], d[:, 1], color=colorsss[1], marker='.', label="abnormal")
#
# plt.legend()
# plt.show()

dataset = 'wadi'
plt.title('ROC Curves for WADI')
with open("/home/lee/PycharmProjects/codes 1.4/roc_"+dataset+ "_taonet", 'rb') as f:
    false_positive_rate, true_positive_rate, auc = pickle.load(f)
    auc = auc*100
print(auc)
# plt.plot(false_positive_rate, true_positive_rate, 'r', label='TAO-Learner AUC = %0.2f'% auc +'%')
plt.plot(false_positive_rate, true_positive_rate, 'r', label='TAO-Learner')
with open("/home/lee/PycharmProjects/codes 1.4/roc_"+dataset+ "_madgan_f1", 'rb') as f:
    false_positive_rate, true_positive_rate, auc = pickle.load(f)
    auc = auc * 100
print(auc)
# plt.plot(false_positive_rate, true_positive_rate, 'b--', label='MAD-GAN AUC = %0.2f'% auc +'%')
plt.plot(false_positive_rate, true_positive_rate, 'b--', label='MAD-GAN')
with open("/home/lee/PycharmProjects/codes 1.4/roc_"+dataset+ "_ocsvm", 'rb') as f:
    false_positive_rate, true_positive_rate, auc = pickle.load(f)
    auc = auc * 100
print(auc)
# plt.plot(false_positive_rate, true_positive_rate, 'y-.', label='OC-SVM AUC = %0.2f'% auc +'%')
plt.plot(false_positive_rate, true_positive_rate, 'y-.', label='OC-SVM')

# plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.grid()
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc='lower right')
plt.show()