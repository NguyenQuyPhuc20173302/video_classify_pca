import model
import pickle
import numpy as np
import matplotlib.pyplot as plt
with open('x_vali.pkl', 'rb') as f:
    x_vali = pickle.load(f)
with open('y_vali.pkl', 'rb') as f:
    y_vali = pickle.load(f)
with open('x_train.pkl', 'rb') as f:
    x_train = pickle.load(f)

x_train, x_vali = model.reduce(x_train, x_vali)

x_vali = x_vali.reshape(12291, 1, 1024)
model = model.lstm_huber((1, 1024), 101, 1)
model.load_weights('weight_pca_huber.hdf5')
model.evaluate(x_vali, y_vali)


'''y_pre = model.predict(x_vali)
# độ chênh lệch giữa thực tế và dự đoán
ketqua = y_vali - y_pre
ketqua = abs(ketqua)
tb = []
for k in ketqua:
    tb.append(np.sum(k))


# mảng đếm số lượng nhãn
count_label = np.zeros(101)

# mảng chứ giá trị trung bình lỗi
loss_ = np.zeros(101)

for j in range(len(y_vali)):
    y = y_vali[j]
    for i in range(len(y)):
        if y[i] == 1:
            count_label[i] += 1
            loss_[i] += tb[j]
            break

label = []
for i in range(101):
    label.append(i)

label = np.array([i for i in label])
plt.bar(label, loss_/count_label, color='green')
plt.title('Trung bình thực tế - dự đoán')
plt.xlabel('tên nhãn')
plt.ylabel('Tổng giá trị chênh lệch giữa (thực tế - dự đoán) / số lượng nhãn')
plt.show()
'''