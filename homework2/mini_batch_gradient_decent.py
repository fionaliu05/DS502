# coding=utf-8
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data_file = 'https://raw.githubusercontent.com/BitTiger-MP/DS502-AI-Engineer/master/DS502-1702/Jason_course/Week2_Codelab2/pima-indians-diabetes.data.csv'
col_name_list = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = pd.read_csv(data_file, names=col_name_list)
array = data_frame.values

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(array[:, :-1])

input_data = rescaledX
target_data = array[:, -1]

num_of_records, n_feature = input_data.shape

x = np.arange(num_of_records)

# 两种终止条件
loop_max = 10000  # 最大迭代次数(防止死循环)
epsilon = 1e-3

# 初始化权值
np.random.seed(0)
w = np.random.randn(input_data.shape[1])

alpha = 0.001  # 步长(注意取值过大会导致振荡,过小收敛速度变慢)
diff = 0.
error = np.zeros(input_data.shape[1])
count = 0  # 循环次数
finish = 0  # 终止标志
error_list = []
# batch_size = 128

n_batch = 100

if num_of_records % n_batch == 0:
    batch_size = int(num_of_records / n_batch)
else:
    batch_size = int(num_of_records / n_batch) + 1

while count < loop_max:
    count += 1

    # 标准梯度下降是在权值更新前对所有样例汇总误差，而随机梯度下降的权值是通过考查某个训练样例来更新的
    # 在标准梯度下降中，权值更新的每一步对多个样例求和，需要更多的计算

    for i in range(n_batch):
        sum_m = np.zeros(n_feature)

        start = i * batch_size
        end = min((i + 1) * batch_size, num_of_records)

        for j in range(start, end):
            dif = (np.dot(w, input_data[j]) - target_data[j]) * input_data[j]
            sum_m = sum_m + dif  # 当alpha取值过大时,sum_m会在迭代过程中会溢出

        w = w - alpha * sum_m  # 注意步长alpha的取值,过大会导致振荡
        error_list.append(np.sum(sum_m)**2)

    # 判断是否已收敛
    if np.linalg.norm(w - error) < epsilon:
        finish = 1
        break
    else:
        error = w

# check with scipy linear regression
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, target_data)
print('intercept = %s slope = %s' % (intercept, slope))

plt.plot(range(len(error_list[0:loop_max])), error_list[0:loop_max])
plt.show()