import torch
import torchvision

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import sys


def get_model_weight(f_name, model, train_data, test_data, device, optim_sgd, epochs):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim_sgd

    test_loss_value = []
    test_acc_value = []

    for epoch in range(epochs):
        print('epoch', epoch + 1)
        for (inputs, labels) in train_data:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        sum_loss = 0.0
        sum_correct = 0
        sum_total = 0

        for (inputs, labels) in test_data:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            _, predicted = outputs.max(1)
            sum_total += labels.size(0)
            sum_correct += (predicted == labels).sum().item()
        print(
            "test loss={},accuracy={}".format(sum_loss * 100 / len(test_data.dataset), float(sum_correct / sum_total)))

    return model, float(sum_correct / sum_total)


from torchvision.models.feature_extraction import create_feature_extractor


def hidden_feature(model, x, lay_name, lay_dict):
    feature_extractor = create_feature_extractor(model, {lay_name: lay_dict})

    feature_dict = feature_extractor(x)

    return feature_dict

from models_Mnist import Cnn10 as md #使用するモデル

model_name='Cnn10'


device=torch.device("cuda")
model=md()
model.to(device)
print(str(model))
'''
f=open('sequence.txt','w')
f.write(str(model))
f.close()
'''

#model=Net()
#model=model.to(device)
criterion=nn.CrossEntropyLoss()
#optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.005)
optimizer=optim.SGD(model.parameters(),lr=0.1)

trans=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,),(0.5,))])

train_data=torchvision.datasets.MNIST(root="mnist_data",train=True,download=True,transform=trans)
train_load=torch.utils.data.DataLoader(train_data,batch_size=100,shuffle=True,num_workers=0)

test_data=torchvision.datasets.MNIST(root="mnist_data",train=False,download=True,transform=trans)
test_load=torch.utils.data.DataLoader(test_data,batch_size=100,shuffle=False,num_workers=0)

test_data_ext=test_data[0][0].view(1,1,28,28)
test_data_ext=test_data_ext.to(device)

#import get_model_weight as gmw

f_name="model_"+model_name+"_"
#f_name="model"

epoch=1
mf=0

use_flag=False

if use_flag:
    import glob
    #モデル転移
    used_model=glob.glob("use_model/*.pth")

    print(used_model)

    path_name=used_model[0]
    model.load_state_dict(torch.load(path_name))
else:
    model, model_acc=get_model_weight(f_name,model,train_load,test_load,device,optimizer,epoch)


#保存関連

'''
f=open('new_model.txt','w')
f.write('epoch:'+str(epoch)+'\n')
f.write('model_pth:'+f_name+str(mf))
f.close()
'''

# import py.get_hidden_layer as ghl

h_log = []
h_logic = {}

# しきいち
hx_a = 0.5

for i in range(10):
    h_logic['label' + str(i)] = []

i = 0

data_ = train_data
# data_=torchvision.datasets.MNIST(root="mnist_data",train=True,download=True,transform=trans)


for (inputs, labels) in data_:
    print(i)
    inputs = inputs[0].view(1, 1, 28, 28).to(device)
    fd = hidden_feature(model, inputs, "pool", "feature")
    print(fd)
    h_x = fd["feature"]
    print(h_x)
    sys.exit()
    h_x = h_x.view(h_x.size()[0], -1)
    # print(model.x_)
    #print(h_x.shape)

    h_x = h_x.to('cpu').detach().numpy().copy()

    print(np.where(h_x >= hx_a, True, False)[0], labels)

    # h_logic['label'+str(labels)].append(np.where(h_x>=0,True,False)[0])
    h_logic['label' + str(labels)].append(np.where(h_x >= hx_a, True, False)[0])

    i += 1

train_logic = h_logic.copy()

# 論理式並べ替え
sor_logic_ind = {}
train_logics = {}
train_logic_nums = {}

for i in range(10):
    print('ind' + str(i))
    h_logic_sub = train_logic['label' + str(i)].copy()
    h_logic_sub = np.array(h_logic_sub)
    logics_ = []
    logic_nums = []
    train_logics['label' + str(i)] = []
    train_logic_nums['label' + str(i)] = []

    while (not h_logic_sub.size == 0):
        logic_ = h_logic_sub[0]
        print(logic_)
        logic_list = np.all(h_logic_sub == logic_, axis=1)
        logic_index = np.where(logic_list == True)
        logic_sum = np.sum(logic_list == True)
        print(logic_sum)
        h_logic_sub = np.delete(h_logic_sub, logic_index, 0)
        logics_.append(logic_)
        logic_nums.append(logic_sum)

    sort_ = np.sort(logic_nums)[::-1]

    sli = np.array([])
    # print(sli)
    num_list = []

    for s in sort_:

        if s in num_list:
            continue

        sli = np.append(sli, np.where(logic_nums == s))
        num_list.append(s)

    for ind in sli:
        ind = int(ind)
        train_logics['label' + str(i)].append(logics_[ind])
        train_logic_nums['label' + str(i)].append(logic_nums[ind])

    sor_logic_ind['label' + str(i)] = sli


print(train_logics['label9'][0],train_logics['label7'][0])
print(train_logic_nums['label9'][0],train_logic_nums['label7'][0])
print(train_logics['label9'][0:10])
print(train_logic_nums['label9'])
sum_=0
for i in range(10):
    for j in train_logic_nums['label'+str(i)]:
        sum_+=j

print(sum_)

# 論理式重複チェック
train_logics_ = train_logics.copy()
train_logic_nums_ = train_logic_nums.copy()

for j in range(10):
    for i in range(len(train_logics_['label' + str(j)])):
        print(j)
        # print(train_logics_['label'+str(j)])
        if len(train_logics_['label' + str(j)]) <= i:
            break
        the_logic = train_logics_['label' + str(j)][i]
        the_num = train_logic_nums_['label' + str(j)][i]
        del_flag = False

        for k in range(10):
            logic_list = np.all(the_logic == train_logics_['label' + str(k)], axis=1)
            # print(logic_list)
            logic_index = np.where(logic_list == True)
            logic_index = list(logic_index[0])
            if not logic_index == []:
                logic_index = logic_index[0]

                if the_num > train_logic_nums_['label' + str(k)][logic_index]:
                    print(the_num, train_logic_nums_['label' + str(k)][logic_index])
                    print(the_logic)
                    print(train_logics_['label' + str(k)][logic_index])
                    train_logics_['label' + str(k)] = np.delete(train_logics_['label' + str(k)], logic_index, 0)
                    train_logic_nums_['label' + str(k)] = np.delete(train_logic_nums_['label' + str(k)], logic_index, 0)
                elif not k == j:
                    print('throw')
                    print(the_num, train_logic_nums_['label' + str(k)][logic_index])
                    print(the_logic)
                    print(train_logics_['label' + str(k)][logic_index])
                    del_flag = True
            if del_flag:
                train_logics_['label' + str(j)] = np.delete(train_logics_['label' + str(j)], i, 0)
                train_logic_nums_['label' + str(j)] = np.delete(train_logic_nums_['label' + str(j)], i, 0)
                break


for i in range(10):
    print(i)
    print(train_logics_['label'+str(i)][0:10])
    print(train_logic_nums['label'+str(i)][0:10])
    print(train_logic_nums_['label'+str(i)][0:10])

# テストデータの論理式
h_log = []
h_logic = {}
hx_a = 0.5

for i in range(10):
    h_logic['label' + str(i)] = []

i = 0

# data_=test_data=torchvision.datasets.MNIST(root="mnist_data",train=False,download=True,transform=trans)
data_ = test_data

for (inputs, labels) in data_:
    print(i)
    inputs = inputs[0].view(1, 1, 28, 28).to(device)
    fd = hidden_feature(model, inputs, "pool", "feature")
    h_x = fd["feature"]
    h_x = h_x.view(h_x.size()[0], -1)

    h_x = h_x.to('cpu').detach().numpy().copy()

    print(np.where(h_x >= hx_a, True, False)[0], labels)

    h_logic['label' + str(labels)].append(np.where(h_x >= hx_a, True, False)[0])

    i += 1

    test_logics = h_logic.copy()

# 論理式正解率
cor_sum = 0
un_exi_sum = 0

for i in range(10):
    logics_sub = list(train_logics_['label' + str(i)].copy())
    cor_ = 0
    un_exi = 0
    if logics_sub != []:
        for log in list(test_logics['label' + str(i)]):
            logic_list = np.all(logics_sub == log, axis=1)

            cor_ += np.sum(logic_list == True)

            if not True in logic_list:
                un_exi += 1

    un_exi_sum += un_exi
    cor_sum += cor_
    print(i, cor_, len(test_logics['label' + str(i)]), cor_sum)
    print(un_exi, un_exi_sum)

#print(logics_sub)
#ルールによる精度
print("model_acc=",model_acc)
print('rule_acc=',cor_sum/10000)
