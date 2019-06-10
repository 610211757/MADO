import numpy as np
import matplotlib.pyplot as plt

x = [24, 30, 36, 42, 48]

y = [0.6163, 0.7386, 0.6208, 0.6128, 0.5736]

plt.plot(x, y)
plt.xticks([24, 30, 36, 42, 48])

# plt.title('WADI')
plt.xlabel('Sequence Length')
plt.ylabel('AUC')

plt.grid(axis="y")
plt.show()

