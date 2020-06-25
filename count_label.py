import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('y_test.pkl', 'rb') as f:
    y_train = pickle.load(f)

num = np.zeros(101)
for y in y_train:
    for i in range(len(y)):
        if y[i] == 1:
            num[i] += 1

label = []
for i in range(101):
    label.append(i)

label = np.array([i for i in label])
plt.bar(label, num, color='green')
plt.title('số lượng nhãn cho tập test')
plt.xlabel('tên nhãn')
plt.ylabel('số lượng')
plt.savefig('numLabel-test.png', format='png')
plt.close()
