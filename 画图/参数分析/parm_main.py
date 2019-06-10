




import numpy as np
import matplotlib.pyplot as plt

ft_size = 24

x_text = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

y = [69.11, 75.48, 69.54, 71.11, 69.12, 70.17, 63.95]

plt.figure(figsize=(16,5))
plt.plot(y,color='b',linestyle=':',marker = 'o',markerfacecolor='r',markersize = 16)
# plt.xlabel(r'$\nu$', fontsize=16)
plt.ylabel('AUC', fontsize=ft_size)

# plt.text(0,0,'Mark')
plt.grid(True)
plt.ylim(50, 80)
plt.xticks(range(np.shape(x_text)[0]),x_text,fontsize=ft_size)
plt.yticks(fontsize=ft_size)

x_text = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]

y = [60.75, 61.06, 62.53, 63.38, 66.01, 69.21, 64.98]

plt.figure(figsize=(16,5))
plt.plot(y,color='b',linestyle=':',marker = 'o',markerfacecolor='r',markersize = 16)
# plt.xlabel(r'$\lambda_{\mathrm{ref}}$', fontsize=16)
plt.ylabel('AUC', fontsize=ft_size)

# plt.text(0,0,'Mark')
plt.grid(True)
plt.ylim(50, 80)
plt.xticks(range(np.shape(x_text)[0]),x_text,fontsize=ft_size)
plt.yticks(fontsize=ft_size)


x_text = [10, 1, 0.1, 0.01, 0.001, 0.0001]

y = [60.98, 63.14, 69.21, 59.12, 58.19, 57.5]

plt.figure(figsize=(16,5))
plt.plot(y,color='b',linestyle=':',marker = 'o',markerfacecolor='r',markersize = 16)
# plt.xlabel(r'$\lambda_{\mathrm{ref}}$', fontsize=16)
plt.ylabel('AUC', fontsize=ft_size)

# plt.text(0,0,'Mark')
plt.grid(True)
plt.ylim(50, 80)
plt.xticks(range(np.shape(x_text)[0]),x_text,fontsize=ft_size)
plt.yticks(fontsize=ft_size)

plt.show()






