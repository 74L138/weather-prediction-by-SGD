import pandas as pd
import torch
import matplotlib.pyplot as plt
import math

data = pd.read_csv('./train.csv', encoding='big5')         #前三列是标题名称，所以是无用的
data = data.iloc[:, 3:]                                    # 去掉前三列，即非数据列,只取有用的数值，其他的全部置零
data[data == 'NR'] = 0                                     # 将rainfall里的NR换成0，即不下雨为0
# print(data)
raw_data = data.to_numpy().astype(float)                   # 下面两行将pandas转换成numpy，再转换成tensor
data_tensor = torch.from_numpy(raw_data)
# print(data_tensor.shape, '\n', data_tensor[0:2])

tensor_split = torch.split(data_tensor, 18, dim=0)   # 由于是18个特征，所以切分尺寸是18
#print(tensor_split[0], len(tensor_split))      # 参数dim表示维度，dim=0则表示按行横着切，dim=1表示按列竖着切
tensor_cat = tensor_split[0]                         # 查看第一行的数据格式，看是否正确
for i in range(len(tensor_split)-1):
    tensor_cat = torch.cat([tensor_cat, tensor_split[i+1]], dim=1)  #拼接，dim=0则表示按行拼接，dim=1表示按列拼接
# print(tensor_cat.shape, len(tensor_cat[0]))#tensor_cat[0]代表的是18行中第一行的数据
#print(tensor_cat[0][0:48])  #输出的是第一行前48个数据

#训练集的每一组数据都应该由两个部分组成：
#前九个小时的数据，共9* 18=162个，reshape到一维行向量，是数据本身，第10个小时的PM2.5值就是目标或者标签。
#采用移位取值的方式来获得数据，步长为1，每次取10个数据，那么可以产生（5760-10）/1 +1=5751组数据。
tensor_t = torch.t(tensor_cat)                  # 转置一下，后面reshape后才是一天的18个特征，接另一天的18个特征...这样组成一个行张量。
tensor_x = torch.empty(len(tensor_t)-10+1, 18*9)
tensor_y = torch.empty(len(tensor_t)-10+1, 1)
# print(tensor_t.shape, len(tensor_t), tensor_x.shape, tensor_y.shape)
for j in range(len(tensor_t)-10+1):
    tensor_x[j] = torch.reshape(tensor_t[j:j+9], [1, -1])    #9行展成一行   reshape(1, -1)：转换成一个行向量。
    tensor_y[j] = tensor_t[j+9][9]                                       #reshape(-1, 1) ：转换成一个列向量。
# print(tensor_x[0], '\n', tensor_y[0:10])

 # pytorch 一维的标注化函数
tensor_norml = torch.nn.BatchNorm1d(162, affine=False, momentum=0.)
tensor_nn_norm = tensor_norml(tensor_x)
# print(tensor_norml, '\n', tensor_nn_norm)

#数据集拆分，80为训练集，20为验证集
x = torch.cat([torch.ones(len(tensor_nn_norm), 1), tensor_nn_norm], dim=1)  # 为1那一列是bias前的系数
w = torch.randn([18*9+1, 1], requires_grad=True)    # 需要求梯度
train_x = x[:math.floor(0.8*len(x))]
train_y = tensor_y[:math.floor(0.8*len(x))]
val_x = x[math.floor(0.8*len(x)):]
val_y = tensor_y[math.floor(0.8*len(x)):]
# print(train_x.shape, train_y.shape, val_x.shape, val_y.shape)
#以上为数据处理部分

#以下为模型训练
lr = 0.02
it_time = 1000
y_train_loss = torch.empty(it_time, 1)
y_val_loss = torch.empty(int(it_time/100), 1)
# 模型、损失函数和优化器实例化
model = torch.nn.Linear(163, 1, bias=False)             # nn.Linear实例化由于前面添加bias进w了，所以这里就不要bias了
loss_function = torch.nn.MSELoss()                      # 使用均方根误差函数
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# 迭代
for iteration in range(it_time):
    # 数据和标签
    inputs = train_x
    targets = train_y
    # 前向传播
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    # 反向传播
    optimizer.zero_grad()       # 梯度清零
    loss.backward()
    optimizer.step()            # 只有用了这个函数模型才会更新
    y_train_loss[iteration] = loss.item()
    if iteration%100 == 0 and iteration != 0:
        val_outputs = model(val_x)
        y_val_loss[int(iteration / 100)] = loss_function(val_outputs, val_y).item()
torch.save(model.state_dict(), 'model.pt')

val_j = range(0, it_time, 100)
plt.figure()
plt.grid()
plt.plot(torch.sqrt(y_train_loss).data.numpy())
plt.plot(val_j, torch.sqrt(y_val_loss).data.numpy(), 'r-')
plt.show()