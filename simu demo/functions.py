'''辅助函数
'''

import numpy as np 
import scipy.io as sio
from scipy.special import gdtr, gdtrix
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
import yaml

# -------------------------------------------------------------------
# DNN模型
# -------------------------------------------------------------------
class MIMOME_IRS_UN_Network(nn.Module):
    def __init__(self, Nt, Nr, Ns, activation_function):
        super(MIMOME_IRS_UN_Network, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(3*(Nt*Nr*(Ns+1)), 64*Ns),
            nn.BatchNorm1d(64*Ns),
            nn.LeakyReLU(0.01),

            nn.Linear(64*Ns, 16*Ns),
            nn.BatchNorm1d(16*Ns),
            nn.LeakyReLU(0.01),

            nn.Linear(16*Ns, 4*Ns),
            nn.BatchNorm1d(4*Ns),
            nn.LeakyReLU(0.01),

            nn.Linear(4*Ns, Ns),
            nn.BatchNorm1d(Ns),
            nn.Sigmoid() if activation_function == 'Sigmoid' else nn.Tanh()
        )

        
    def forward(self, x):
        return self.main(x)


# -------------------------------------------------------------------
# CNN模型: (3*Nr,Nt,Ns+1)
# -------------------------------------------------------------------
class MIMOME_IRS_CNN_Network(nn.Module):
    def __init__(self, Nt, Nr, Ns, hidden_dim1, hidden_dim2, activation_function):
        super(MIMOME_IRS_CNN_Network, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(Ns+1, hidden_dim1, kernel_size=2, stride=1),
            nn.BatchNorm2d(hidden_dim1),
            nn.LeakyReLU(0.01),
            # nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=2, stride=1),
            nn.BatchNorm2d(hidden_dim2),
            nn.LeakyReLU(0.01),
            # nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.fc1 = nn.Sequential(
            nn.Linear((3*Nr-2)*(Nt-2)*hidden_dim2, 128*Ns),
            nn.BatchNorm1d(128*Ns),
            nn.LeakyReLU(0.01),
            )

        self.fc2 = nn.Sequential(
            nn.Linear(128*Ns, 16*Ns),
            nn.BatchNorm1d(16*Ns),
            nn.LeakyReLU(0.01),
            )
        
        self.fc3 = nn.Sequential(
            nn.Linear(16*Ns, Ns),
            nn.BatchNorm1d(Ns),
            # nn.Sigmoid()
            nn.Sigmoid() if activation_function == 'Sigmoid' else nn.Tanh()
            )
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out

# -------------------------------------------------------------------
# CNN3模型: (2*Nr,Nt,Ns+1)
# -------------------------------------------------------------------
class MIMOME_IRS_CNN3_Network(nn.Module):
    def __init__(self, Nt, Nr, Ns, hidden_dim1, hidden_dim2, activation_function):
        super(MIMOME_IRS_CNN3_Network, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(Ns+1, hidden_dim1, kernel_size=2, stride=1),
            nn.BatchNorm2d(hidden_dim1),
            nn.LeakyReLU(0.01),
            # nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=2, stride=1),
            nn.BatchNorm2d(hidden_dim2),
            nn.LeakyReLU(0.01),
            # nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.fc1 = nn.Sequential(
            nn.Linear((2*Nr-2)*(Nt-2)*hidden_dim2, 128*Ns),
            nn.BatchNorm1d(128*Ns),
            nn.LeakyReLU(0.01),
            )

        self.fc2 = nn.Sequential(
            nn.Linear(128*Ns, 16*Ns),
            nn.BatchNorm1d(16*Ns),
            nn.LeakyReLU(0.01),
            )
        
        self.fc3 = nn.Sequential(
            nn.Linear(16*Ns, Ns),
            nn.BatchNorm1d(Ns),
            # nn.Sigmoid()
            nn.Sigmoid() if activation_function == 'Sigmoid' else nn.Tanh()
            )
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out

# -------------------------------------------------------------------
# 生成训练集和验证集数据
# -------------------------------------------------------------------
def Gen_Train_Data(mode, TotalNum, TrainNum, batch_size, num_workers, pin_memory, Nt, Nr, Ns, sigma_e, alpha0, beta0):
    '''生成训练集需要的数据

        Nt: 发送端天线
        Nr: 接收端天线
        Ns: IRS元件数

    '''
    if mode == 'UN':
        # random.randn产生(0,1)正态分布,还有个random.normal也可以
        Gr_orign = np.sqrt(1/2)*(np.random.randn(TotalNum, Nr, Ns).astype('float32') + 1j*np.random.randn(TotalNum, Nr, Ns).astype('float32'))
        Hb_orign = np.sqrt(1/2)*(np.random.randn(TotalNum, Nr, Nt).astype('float32') + 1j*np.random.randn(TotalNum, Nr, Nt).astype('float32'))
        H_orign = np.sqrt(1/2)*(np.random.randn(TotalNum, Ns, Nt).astype('float32') + 1j*np.random.randn(TotalNum, Ns, Nt).astype('float32'))

        Gr = Gr_orign.reshape(TotalNum, Nr*Ns)
        Hb = Hb_orign.reshape(TotalNum, Nr*Nt)
        H = H_orign.reshape(TotalNum, Ns*Nt)
        alpha = alpha0*np.ones((TotalNum, 1)).astype('float32')
        beta = beta0*np.ones((TotalNum, 1)).astype('float32')
        # 功率-4～21dB
        Power = np.random.randint(-4, 21, size=[TotalNum, 1])
        # Power = 10 * np.ones((TotalNum, 1)).astype('float32')
        SNR = np.power(10, (Power - sigma_e) / 10).astype('float32')
        # Rs = 1.5 * np.ones((TotalNum, 1)).astype('float32')
        Rs = np.random.randint(3, 10, size=[TotalNum, 1]) / 2.0

        origin_data = np.concatenate((alpha, beta, SNR, Rs, H, Gr, Hb), axis=-1)
        train_label = torch.from_numpy(origin_data).to(torch.complex64)

        train_origin_data = np.concatenate((np.real(Hb), np.imag(Hb), np.abs(Hb)), axis=-1)
        for i in range(Ns):
            F = np.expand_dims(Gr_orign[:,:,i], axis=-1) @ np.expand_dims(H_orign[:,i,:], axis=1)
            F = F.reshape(TotalNum, Nr*Nt)
            train_origin_data = np.concatenate((train_origin_data, np.real(F), np.imag(F), np.abs(F)), axis=-1)

        train_data = torch.from_numpy(train_origin_data).to(torch.float32)

        train_dataset = TensorDataset(train_data[:TrainNum], train_label[:TrainNum])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

        valid_dataset = TensorDataset(train_data[TrainNum:], train_label[TrainNum:])
        # valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

        del F, Gr_orign, Hb_orign, H_orign, origin_data, train_origin_data

        return train_dataloader, valid_dataloader
    
    elif mode == 'CNN':
        Gr_orign = np.sqrt(1/2)*(np.random.randn(TotalNum, Nr, Ns).astype('float32') + 1j*np.random.randn(TotalNum, Nr, Ns).astype('float32'))
        Hb_orign = np.sqrt(1/2)*(np.random.randn(TotalNum, Nr, Nt).astype('float32') + 1j*np.random.randn(TotalNum, Nr, Nt).astype('float32'))
        H_orign = np.sqrt(1/2)*(np.random.randn(TotalNum, Ns, Nt).astype('float32') + 1j*np.random.randn(TotalNum, Ns, Nt).astype('float32'))

        Gr = Gr_orign.reshape(TotalNum, Nr*Ns)
        Hb = Hb_orign.reshape(TotalNum, Nr*Nt)
        H = H_orign.reshape(TotalNum, Ns*Nt)
        alpha = alpha0*np.ones((TotalNum, 1)).astype('float32')
        beta = beta0*np.ones((TotalNum, 1)).astype('float32')
        # 功率-4～21dB
        Power = np.random.randint(-4, 21, size=[TotalNum, 1])
        # Power = 10 * np.ones((TotalNum, 1)).astype('float32')
        SNR = np.power(10, (Power - sigma_e) / 10).astype('float32')
        # Rs = 1.5 * np.ones((TotalNum, 1)).astype('float32')
        Rs = np.random.randint(3, 10, size=[TotalNum, 1]) / 2.0

        origin_data = np.concatenate((alpha, beta, SNR, Rs, H, Gr, Hb), axis=-1)
        train_label = torch.from_numpy(origin_data).to(torch.complex64)

        # Nr*Nt*(Ns+1)维度
        F = np.empty((TotalNum, Nr, Nt, Ns), dtype=complex)
        for i in range(Ns):
            F[:,:,:,i] = np.expand_dims(Gr_orign[:,:,i], axis=-1) @ np.expand_dims(H_orign[:,i,:], axis=1)

        F_cascade = np.concatenate((F, np.expand_dims(Hb_orign, axis=-1)), axis=-1)

        # 按Nr维度拼接, (3*Nr, Nt, Ns+1)
        train_origin_data = np.concatenate((np.real(F_cascade), np.imag(F_cascade), np.abs(F_cascade)), axis=1)
        # train_origin_data = F_cascade

        # PyTorch的维度应该是 Channel * height * width
        train_data0 = torch.from_numpy(train_origin_data).to(torch.float32)
        # train_data0 = torch.from_numpy(train_origin_data).to(torch.complex64).to(args.device)
        train_data = train_data0.permute(0, 3, 1, 2)

        train_dataset = TensorDataset(train_data[:TrainNum], train_label[:TrainNum])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

        valid_dataset = TensorDataset(train_data[TrainNum:], train_label[TrainNum:])
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

        del F, Gr_orign, Hb_orign, H_orign, F_cascade, origin_data, train_origin_data

        return train_dataloader, valid_dataloader

    elif mode == 'CNN3':
        Gr_orign = np.sqrt(1/2)*(np.random.randn(TotalNum, Nr, Ns).astype('float32') + 1j*np.random.randn(TotalNum, Nr, Ns).astype('float32'))
        Hb_orign = np.sqrt(1/2)*(np.random.randn(TotalNum, Nr, Nt).astype('float32') + 1j*np.random.randn(TotalNum, Nr, Nt).astype('float32'))
        H_orign = np.sqrt(1/2)*(np.random.randn(TotalNum, Ns, Nt).astype('float32') + 1j*np.random.randn(TotalNum, Ns, Nt).astype('float32'))

        Gr = Gr_orign.reshape(TotalNum, Nr*Ns)
        Hb = Hb_orign.reshape(TotalNum, Nr*Nt)
        H = H_orign.reshape(TotalNum, Ns*Nt)
        alpha = alpha0*np.ones((TotalNum, 1)).astype('float32')
        beta = beta0*np.ones((TotalNum, 1)).astype('float32')
        # 功率-4～21dB
        Power = np.random.randint(-4, 21, size=[TotalNum, 1])
        # Power = 10 * np.ones((TotalNum, 1)).astype('float32')
        SNR = np.power(10, (Power - sigma_e) / 10).astype('float32')
        # Rs = 1.5 * np.ones((TotalNum, 1)).astype('float32')
        Rs = np.random.randint(3, 10, size=[TotalNum, 1]) / 2.0

        origin_data = np.concatenate((alpha, beta, SNR, Rs, H, Gr, Hb), axis=-1)
        train_label = torch.from_numpy(origin_data).to(torch.complex64)

        # Nr*Nt*(Ns+1)维度
        F = np.empty((TotalNum, Nr, Nt, Ns), dtype=complex)
        for i in range(Ns):
            F[:,:,:,i] = np.expand_dims(Gr_orign[:,:,i], axis=-1) @ np.expand_dims(H_orign[:,i,:], axis=1)

        F_cascade = np.concatenate((F, np.expand_dims(Hb_orign, axis=-1)), axis=-1)

        # 按Nr维度拼接, (3*Nr, Nt, Ns+1)
        train_origin_data = np.concatenate((np.real(F_cascade), np.imag(F_cascade)), axis=1)
        # train_origin_data = F_cascade

        # PyTorch的维度应该是 Channel * height * width
        train_data0 = torch.from_numpy(train_origin_data).to(torch.float32)
        # train_data0 = torch.from_numpy(train_origin_data).to(torch.complex64).to(args.device)
        train_data = train_data0.permute(0, 3, 1, 2)

        train_dataset = TensorDataset(train_data[:TrainNum], train_label[:TrainNum])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

        valid_dataset = TensorDataset(train_data[TrainNum:], train_label[TrainNum:])
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

        del F, Gr_orign, Hb_orign, H_orign, F_cascade, origin_data, train_origin_data

        return train_dataloader, valid_dataloader

def Get_Test_Data(Network_mode, TestNum, Nt, Nr, Ns, Gr, H, Hb, Hb_test):
    '''对测试数据进行预处理
    '''
    if Network_mode == 'UN':
        test_origin_data = np.concatenate((np.real(Hb_test), np.imag(Hb_test), np.abs(Hb_test)), axis=-1)
        for i in range(Ns):
            F = np.expand_dims(Gr[:,:,i], axis=-1) @ np.expand_dims(H[:,i,:], axis=1)
            F = F.reshape(TestNum, Nr*Nt)
            test_origin_data = np.concatenate((test_origin_data, np.real(F), np.imag(F), np.abs(F)), axis=-1)

        test_data = torch.from_numpy(test_origin_data).to(torch.float32)

        return test_data

    elif Network_mode == 'CNN':
        # (3*Nr, Nt, Ns+1)
        F = np.empty((TestNum, Nr, Nt, Ns), dtype=complex)
        for i in range(Ns):
            F[:,:,:,i] = np.expand_dims(Gr[:,:,i], axis=-1) @ np.expand_dims(H[:,i,:], axis=1)

        F_cascade = np.concatenate((F, np.expand_dims(Hb, axis=-1)), axis=-1)
        test_origin_data = np.concatenate((np.real(F_cascade), np.imag(F_cascade), np.abs(F_cascade)), axis=1)
        test_data0 = torch.from_numpy(test_origin_data).to(torch.float32)
        test_data = test_data0.permute(0, 3, 1, 2)

        return test_data

    elif Network_mode == 'CNN3':
        # (3*Nr, Nt, Ns+1)
        F = np.empty((TestNum, Nr, Nt, Ns), dtype=complex)
        for i in range(Ns):
            F[:,:,:,i] = np.expand_dims(Gr[:,:,i], axis=-1) @ np.expand_dims(H[:,i,:], axis=1)

        F_cascade = np.concatenate((F, np.expand_dims(Hb, axis=-1)), axis=-1)
        test_origin_data = np.concatenate((np.real(F_cascade), np.imag(F_cascade)), axis=1)
        test_data0 = torch.from_numpy(test_origin_data).to(torch.float32)
        test_data = test_data0.permute(0, 3, 1, 2)

        return test_data


# -------------------------------------------------------------------
# 早停
# -------------------------------------------------------------------
class EarlyStopping():
    def __init__(self,patience=20,verbose=False,delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self,val_loss,model,path):
        print("val_loss={:.4f}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,model,path)
        elif score < self.best_score+self.delta:
            self.counter+=1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter>=self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,model,path)
            self.counter = 0
    def save_checkpoint(self,val_loss,model,path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.4f} ===> {val_loss:.4f}).  Saving model to ' + path)
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


# -------------------------------------------------------------------
# 损失函数
# -------------------------------------------------------------------
class MyLossFunction(nn.Module):
    def __init__(self, Nt, Nr, Ns, device, activation_function):
        super(MyLossFunction, self).__init__()
        self.Nt = Nt
        self.Nr = Nr
        self.Ns = Ns
        self.device = device
        self.factor = 2*torch.pi if activation_function == 'Sigmoid' else torch.pi
    
    def forward(self, output, target):
        alpha = target[:, 0][0]
        beta = target[:, 1][0]
        # alpha0 = torch.unsqueeze(target[:, 0], dim=-1)
        # beta0 = torch.unsqueeze(target[:, 1], dim=-1)
        SNR = torch.unsqueeze(target[:, 2], dim=-1)
        Rs = torch.unsqueeze(target[:, 3], dim=-1)

        H0 = target[:, 4:4+self.Ns*self.Nt]
        Gr0 = target[:, 4+self.Ns*self.Nt:4+self.Ns*self.Nt+self.Nr*self.Ns]
        Hb0 = target[:, 4+self.Ns*self.Nt+self.Nr*self.Ns:4+self.Ns*self.Nt+self.Nr*self.Ns+self.Nr*self.Nt]

        H = H0.reshape(-1, self.Ns, self.Nt)
        Gr = Gr0.reshape(-1, self.Nr, self.Ns)
        Hb = Hb0.reshape(-1, self.Nr, self.Nt)
        # alpha = alpha0.reshape(-1, 1, 1)
        # beta = beta0.reshape(-1, 1, 1)

        Phase = output.float() * self.factor
        theta = torch.complex(torch.cos(Phase), torch.sin(Phase))
        Phi = torch.diag_embed(theta)
        # theta1 = torch.cos(Phase) + 1j*torch.sin(Phase)
        # theta2 = np.cos(Phase) + 1j*np.sin(Phase)
        
        '''
        Beam_real = output[:, Ns:Ns+Nt].float()
        Beam_imag = output[:, Ns+Nt:].float()
        # Beam0 = tf.cast(tf.complex(Beam_real, Beam_imag), tf.complex64)
        Beam0 = torch.complex(Beam_real, Beam_imag)
        Beam = torch.unsqueeze(Beam0, dim=-1)
        '''
        
        t = ((1-torch.pow(2, Rs)) / SNR).reshape(-1, 1, 1)
        EyeMatrix0 = torch.ones(H.shape[0], self.Nt).cuda(device=self.device)
        # 在loss里创建的变量，要设置到GPU上运行
        EyeMatrix = torch.diag_embed(EyeMatrix0).cuda(device=self.device)

        # A1 = torch.transpose((alpha*Hb + Gr @ Phi @ H).conj(), dim0=1, dim1=2) @ (alpha*Hb + Gr @ Phi @ H) +\
        #     t*EyeMatrix
        # A2 = torch.transpose(H.conj(), dim0=1, dim1=2) @ H +\
        #     beta**2 * EyeMatrix
        
        A1 = torch.transpose((Hb + Gr @ Phi @ H).conj(), dim0=1, dim1=2) @ (Hb + Gr @ Phi @ H) +\
            t*EyeMatrix
        A2 = alpha**2 * torch.transpose(H.conj(), dim0=1, dim1=2) @ H +\
            beta**2 * EyeMatrix

        XX = torch.linalg.inv(A2) @ A1
        eig_value, eig_vector = torch.linalg.eig(XX)
        # eig_value2, eig_vector2 = np.linalg.eig(XX.numpy())

        # 取出来最大特征值对应的特征向量
        _, indices = torch.max(eig_value.abs(), dim=-1, keepdim=True)
        Beam = eig_vector.gather(-1, indices.unsqueeze(-1).expand(-1, self.Nt, 1))
        
        Cm = torch.log2(1 + SNR*torch.norm((Hb + Gr @ Phi @ H) @ Beam, dim=1)**2)
        phi_1 = (torch.pow(2, (Cm-Rs)) - 1) / SNR
        z0 = phi_1 / (beta**2 + alpha**2 * torch.norm(Phi @ H @ Beam, dim=1)**2)

        loss = -torch.mean(z0).float()

        return loss

# -------------------------------------------------------------------
# 加载全局配置
# -------------------------------------------------------------------
def load_config(config_file):
    f = open(config_file, 'r', encoding='utf-8')
    a = f.read()
    result = yaml.load(a, Loader=yaml.FullLoader)
    return result


