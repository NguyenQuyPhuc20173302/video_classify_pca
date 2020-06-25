import numpy as np
import pickle as p
import matplotlib.pyplot as plt
with open('y_train.pkl', 'rb') as f:
    y_train = p.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = p.load(f)
with open('y_vali.pkl', 'rb') as f:
    y_vali = p.load(f)
num_label = np.zeros(101)


def tk(y_train):
    for y in y_train:
        for i in range(len(y)):
            if y[i] == 1:
                num_label[i] += 1
                break


tk(y_train)
tk(y_test)
tk(y_vali)
print(num_label.max())
label = []
for i in range(101):
    label.append(i)

label = np.array([i for i in label])
plt.bar(label, num_label, color='green')
plt.title('số lượng nhãn')
plt.xlabel('tên nhãn')
plt.ylabel('số lượng')
plt.show()