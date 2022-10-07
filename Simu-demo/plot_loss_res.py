#%%

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator 
#%%

dataFileCNN = './result/loss_CNN.mat'
dataFileCNN3 = './result/loss_CNN3.mat'
dataFileUN = './result/loss_UN.mat'

res_loss_cnn = sio.loadmat(dataFileCNN)
res_loss_un = sio.loadmat(dataFileUN)
res_loss_cnn3 = sio.loadmat(dataFileCNN3)
# res_loss_cnn

# %%
train_epochs_loss_cnn = res_loss_cnn['train_epochs_loss']
valid_epochs_loss_cnn = res_loss_cnn['valid_epochs_loss']

train_epochs_loss_cnn = np.squeeze(train_epochs_loss_cnn)
valid_epochs_loss_cnn = np.squeeze(valid_epochs_loss_cnn)

train_epochs_loss_un = res_loss_un['train_epochs_loss']
valid_epochs_loss_un = res_loss_un['valid_epochs_loss']

train_epochs_loss_un = np.squeeze(train_epochs_loss_un)
valid_epochs_loss_un = np.squeeze(valid_epochs_loss_un)

train_epochs_loss_cnn3 = res_loss_cnn3['train_epochs_loss']
valid_epochs_loss_cnn3 = res_loss_cnn3['valid_epochs_loss']

train_epochs_loss_cnn3 = np.squeeze(train_epochs_loss_cnn3)
valid_epochs_loss_cnn3 = np.squeeze(valid_epochs_loss_cnn3)


# plt.figure(figsize=(12,4))
plt.figure()

# plt.subplot(121)
plt.plot(train_epochs_loss_un[1:30],'g-.',label="train loss FCN")
plt.plot(valid_epochs_loss_un[1:30],'g-.+',label="valid loss FCN")

plt.plot(train_epochs_loss_cnn[1:30],'k-',label="train loss CNN")
plt.plot(valid_epochs_loss_cnn[1:30],'k-+',label="valid loss CNN",markersize='8')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('./result/compare_network_architecture.pdf')


plt.figure()
plt.plot(train_epochs_loss_cnn3[1:30],'r--',label="train loss without the abs value")
plt.plot(valid_epochs_loss_cnn3[1:30],'r--+',label="valid loss without the abs value",markersize='8')

plt.plot(train_epochs_loss_cnn[1:30],'k-',label="train loss")
plt.plot(valid_epochs_loss_cnn[1:30],'k-+',label="valid loss",markersize='8')

plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('./result/compare_input_features.pdf')
plt.show()


# %%





