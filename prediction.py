import pandas as pd
import csv
import torch
test_data = pd.read_csv('./test.csv', encoding='big5', header=None)
test_data = test_data.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_raw_data = test_data.to_numpy().astype(float)
test_tensor = torch.from_numpy(test_raw_data)

test_x = torch.zeros(240, 18*9)
test_split = torch.split(test_tensor, 18, dim=0)
# print(len(test_split))
for i in range(len(test_split)):
    test_x[i] = torch.reshape(torch.t(test_split[i]), [1, -1])    # 注意此处要转置，因为处理训练集的时候转置了，而且转置之后才是比较和逻辑的。
# print(test_x[0])
norm = torch.nn.BatchNorm1d(162, affine=False, momentum=0.)      # 标准化
test_x = norm(test_x)

x = torch.cat([torch.ones(len(test_x), 1), test_x], dim=1)
# print(x)
# print(x.shape)

#w = torch.load('model.pt')
#y_pred = torch.mm(x, w)

# 用nn模型预测
model = torch.nn.Linear(163, 1, bias=False)             # nn.Linear实例化由于前面添加bias进w了，所以这里就不要bias了
model.load_state_dict(torch.load('model.pt'))
y_pred = model(x)
#print(y_pred)

y_pred_sq = torch.squeeze(y_pred)  # 压缩维度
with open('predict.csv', mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    header = ['id', 'PM2.5']
    csv_writer.writerow(header)
    for j in range(len(y_pred_sq)):
        if int(y_pred_sq[j]) < 0:
            row = ['id_'+str(j), '0']
        else:
            row = ['id_' + str(j), str(int(y_pred_sq[j]))]
        csv_writer.writerow(row)