#%% 
import numpy as np 
import scipy.io as sio
from scipy.special import gdtr, gdtrix
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset

from functions import MyLossFunction, EarlyStopping, MIMOME_IRS_UN_Network, MIMOME_IRS_CNN_Network, MIMOME_IRS_CNN3_Network
from functions import load_config, Gen_Train_Data
import yaml
import matplotlib.pyplot as plt

import random
seed = 2
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

#%%
# ==================
# 加载参数
# ==================
cfg_file = 'cfg_Nt=4_Nr=2_Ns=16.yaml'
args = load_config('./cfg_file/' + cfg_file)

# 神经网络参数
Network_mode = args['Network_mode']
epochs = args['epochs']
learning_rate = args['learning_rate']
batch_size = args['batch_size']
num_workers = args['num_workers']
pin_memory = args['pin_memory']
CNN_hidden_dim1 = args['CNN_hidden_dim1']
CNN_hidden_dim2 = args['CNN_hidden_dim2']

TotalNum = args['TotalNum']
TrainNum = int(0.8*TotalNum)
# device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),]
device, = [torch.device(args['device'] if torch.cuda.is_available() else "cpu"),]
earlystopping_patience = args['earlystopping_patience']
reducelr_patience = args['reducelr_patience']
activation_function = args['activation_function']

# 天线参数
Nt, Nr, Ne, Ns = args['Nt'], args['Nr'], args['Ne'], args['Ns']
sigma_e, alpha, beta = args['sigma_e'], args['alpha'], args['beta']

# 模型参数保存路径
model_path = args['model_path'] + '/PyTorch_'+ Network_mode +'_best_' + cfg_file[:-5] + '.pth'
result_path = args['result_path'] + '/'+ Network_mode +'_Method_' + cfg_file[:-5] + '.mat'
dataFile = args['dataFile']

#%%
# ==================
# 生成测试集和验证集
# ==================
train_dataloader, valid_dataloader = Gen_Train_Data(Network_mode, TotalNum, TrainNum, batch_size, num_workers, pin_memory, Nt, Nr, Ns, sigma_e, alpha, beta)

#%%
# ==================
# 构建网络
# ==================
if Network_mode == 'UN':
    model = MIMOME_IRS_UN_Network(Nt, Nr, Ns, activation_function).to(device)
elif Network_mode == 'CNN':
    model = MIMOME_IRS_CNN_Network(Nt, Nr, Ns, CNN_hidden_dim1, CNN_hidden_dim2, activation_function).to(device)
elif Network_mode == 'CNN3':
    model = MIMOME_IRS_CNN3_Network(Nt, Nr, Ns, CNN_hidden_dim1, CNN_hidden_dim2, activation_function).to(device)

print(model)
criterion = MyLossFunction(Nt, Nr, Ns, device, activation_function)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []

# 设置早停
early_stopping = EarlyStopping(patience=earlystopping_patience, verbose=True)
# 自动调整学习率
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.33, threshold=0, patience=reducelr_patience, verbose=True)
#%%
# ==================
# 训练模型
# ==================
for epoch in range(epochs):
    start = time.time()
    model.train()
    train_epoch_loss = []
    for idx,(data_x,data_y) in enumerate(train_dataloader,0):
        data_x = data_x.to(device)
        data_y = data_y.to(device)
        outputs = model(data_x)
        optimizer.zero_grad()
        loss = criterion(outputs,data_y)
        loss.backward()
        optimizer.step()
        train_epoch_loss.append(loss.item())
        train_loss.append(loss.item())
        # if idx%(len(train_dataloader)//2)==0:
        #     print("epoch={}/{},{}/{} of train, loss={:.4f}".format(
        #         epoch, args.epochs, idx, len(train_dataloader),loss.item()))
    
    end = time.time()
    train_epochs_loss.append(np.average(train_epoch_loss))
    print('Epoch:{}/{} | train_loss:{:.4f} | time:{:.4f}s | learning rate:{}'.format(epoch+1, epochs, train_epochs_loss[-1],(end-start), optimizer.state_dict()['param_groups'][0]['lr']))
    
    #=====================valid============================
    with torch.no_grad():
        model.eval()
        valid_epoch_loss = []
        for idx,(data_x,data_y) in enumerate(valid_dataloader,0):
            data_x = data_x.to(device)
            data_y = data_y.to(device)
            outputs = model(data_x)
            loss = criterion(outputs,data_y)
            valid_epoch_loss.append(loss.item())
            valid_loss.append(loss.item())
    valid_epochs_loss.append(np.average(valid_epoch_loss))

    #==================early stopping======================
    early_stopping(valid_epochs_loss[-1],model=model,path=model_path)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    #====================adjust lr=========================
    ''' 
    1. 通过设置的指标自动调整.
    '''
    scheduler.step(valid_epochs_loss[-1])
    
    '''
    2. 手动的方式, 分段调整学习率.
    '''
    # lr_adjust = {
    #         2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
    #         10: 5e-7, 15: 1e-7, 20: 5e-8
    #     }
    # if epoch in lr_adjust.keys():
    #     lr = lr_adjust[epoch]
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    #     print('Updating learning rate to {}'.format(lr))

#%%

res_loss_path = './result/' + 'loss_' + Network_mode + '.mat'
sio.savemat(res_loss_path, {'train_epochs_loss':train_epochs_loss, 'valid_epochs_loss':valid_epochs_loss})
# 绘图
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(train_loss[:])
plt.title("train_loss")
plt.subplot(122)
plt.plot(train_epochs_loss[1:],'-o',label="train_loss")
plt.plot(valid_epochs_loss[1:],'-o',label="valid_loss")
plt.title("epochs_loss")
plt.legend()
plt.show()
