import argparse
import numpy as np
import os
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.utils.data import DataLoader, Dataset
import cvxpy as cp
from UnmixingUtils import UnmixingUtils
class L1NMF_Net(nn.Module):
    def __init__(self, layerNum, M, A):
        super(L1NMF_Net, self).__init__()
        R = np.size(M, 1)
        eig, _ = np.linalg.eig(M.T @ M)
        eig += 0.1
        L = 1 / np.max(eig)
        theta = np.ones((1, R)) * 0.01 * L
        # Endmember
        eig, _ = np.linalg.eig(A @ A.T)
        eig += 0.1
        L2 = np.max(eig)
        L2 = 1 / L2

        self.p = nn.ParameterList()
        self.L = nn.ParameterList()
        self.theta = nn.ParameterList()
        self.L2 = nn.ParameterList()
        self.W_a = nn.ParameterList()
        self.layerNum = layerNum
        temp = self.calW(M)
        for k in range(self.layerNum):
            self.L.append(nn.Parameter(torch.FloatTensor([L])))
            self.L2.append(nn.Parameter(torch.FloatTensor([L2])))
            self.theta.append(nn.Parameter(torch.FloatTensor(theta)))
            self.p.append(nn.Parameter(torch.FloatTensor([0.5])))
            self.W_a.append(nn.Parameter(torch.FloatTensor(temp)))
        self.layerNum = layerNum
    def forward(self, X, _M, _A):
        self.W_m = torch.FloatTensor(_A)
        M = list()
        M.append(torch.FloatTensor(_M))
        A = list()
        A.append(torch.FloatTensor(_A.T))
        for k in range(self.layerNum):
            theta = self.theta[k].repeat(A[-1].size(1), 1).T
            T = M[-1].mm(A[-1]) - X
            _A = A[-1] - self.L[k]*self.W_a[k].T.mm(T)
            _A = self.sum2one(F.relu(self.self_active(_A, self.p[k], theta)))
            A.append(_A)
            T = M[-1].mm(A[-1]) - X
            _M = M[-1] - T.mm(self.L2[k] * self.W_m)
            _M = F.relu(_M)
            M.append(_M)
        return M, A
    def half_thresholding(self, z_hat, mu):
        c=pow(54,1/3)/4
        tau=z_hat.abs()-c*pow(mu,2/3)
        v=z_hat
        ind=tau>0
        v[ind]=2/3*z_hat[ind]*(1+torch.cos(2*math.pi/3-2/3*torch.acos(mu[ind]/8*pow(z_hat[ind].abs()/3,-1.5))))
        v[tau<0]=0
        return v
    def soft_thresholding(self, z_hat, mu):
        return z_hat.sign() * F.relu(z_hat.abs() - mu)
    def self_active(self, x, p, lam):
        tau=pow(2*(1-p)*lam,1/(2-p))+p*lam*pow(2*lam*(1-p), (p-1)/(2-p))
        v = x
        ind = (x-tau) > 0
        ind2=(x-tau)<=0
        v[ind]=x[ind].sign() * (x[ind].abs() - p * lam[ind] * pow(x[ind].abs(), p - 1))
        v[ind2]=0
        v[v>1]=1
        return v
    def calW(self,D):
        (m,n)=D.shape
        W = cp.Variable(shape=(m, n))
        obj = cp.Minimize(cp.norm(W.T @ D, 'fro'))
        # Create two constraints.
        constraint = [cp.diag(W.T @ D) == 1]
        prob = cp.Problem(obj, constraint)
        result = prob.solve(solver=cp.SCS, max_iters=1000)
        print('residual norm {}'.format(prob.value))
        # print(W.value)
        return W.value
    def sum2one(self, Z):
        temp = Z.sum(0)
        temp = temp.repeat(Z.size(0), 1) + 0.0001
        return Z / temp
class RandomDataset(Dataset):
    def __init__(self, data, label, length):
        self.data = data
        self.len = length
        self.label = label

    def __getitem__(self, item):
        return torch.Tensor(self.data[:,item]).float(), torch.Tensor(self.label[:,item]).float()

    def __len__(self):
        return self.len


def prepare_data(dataFile):
    data = scio.loadmat(dataFile)
    X = data['x_n']
    A = data['A']
    s = data['s']
    return X, A, s

def prepare_train(X, s, trainFile):
    train_index = scio.loadmat(trainFile)
    train_index = train_index['train']
    train_index=train_index-1
    train_data = np.squeeze(X[:, train_index])
    train_labels = np.squeeze(s[:, train_index])
    nrtrain = np.size(train_index, 1)
    return train_data, train_labels, nrtrain




def prepare_init(initFile):
    init = scio.loadmat(initFile)
    A0 = init['Cn']
    S0 = init['o'][0, 0]['S']
    return A0, S0


def set_param(layerNum, lr, lrD, batch_size=4096):
    parser = argparse.ArgumentParser(description="LISTA-Net")
    parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
    parser.add_argument('--end_epoch', type=int, default=500, help='epoch number of end training')
    parser.add_argument('--layer_num', type=int, default=layerNum, help='phase number of ISTA-Net')
    parser.add_argument('--learning_rate_decoder', type=float, default=lrD, help='learning rate for decoder')
    parser.add_argument('--learning_rate', type=float, default=lr, help='learning rate')
    parser.add_argument('--batch_size', type=float, default=batch_size, help='batch size')
    parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
    parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
    parser.add_argument('--log_dir', type=str, default='log', help='log directory')
    args = parser.parse_args()
    return args
def train(lrD,layerNum, lr, train_data, test_data, nrtrain, A0, S0, X, A, s, SNR):
    batch_size = nrtrain
    args = set_param(layerNum, lr, lrD,batch_size=batch_size)
    model_dir = "./%s/SNR_%sSNMF_layer_%d_lr_%.8f_lrD_%.8f" % (
        args.model_dir, SNR, args.layer_num, args.learning_rate, args.learning_rate_decoder)
    log_file_name = "./%s/SNR_%sSNMF_layer_%d_lr_%.8f_lrD_%.8f.txt" % (
        args.log_dir, SNR, args.layer_num, args.learning_rate, args.learning_rate_decoder)
    model = L1NMF_Net(args.layer_num, A0, S0)
    criterion = nn.MSELoss(reduction='sum')
    trainloader = DataLoader(dataset=RandomDataset(train_data, test_data, nrtrain), batch_size=args.batch_size,
                             num_workers=0,
                             shuffle=False)
    learning_rate = args.learning_rate
    learning_rate_decoder=args.learning_rate_decoder
    opt = optim.Adam([{'params': [L_a for L_a in model.L] + [p for p in model.p]},
                      {'params': [L_b for L_b in model.L2] + [W_a_ for W_a_ in model.W_a] + [the for the in
                                                                       model.theta],
                       'lr': learning_rate_decoder}],
                     lr=learning_rate, weight_decay=0.001, betas=(0.9, 0.9))
    start_epoch = args.start_epoch
    end_epoch = args.end_epoch
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    running_loss = 0.0
    last_loss=1
    for epoch_i in range(start_epoch + 1, end_epoch + 1):
        if epoch_i <= 5 and epoch_i % 2 == 0:
            learning_rate = learning_rate / 25
            opt = optim.Adam([{'params': [L_a for L_a in model.L] + [p for p in model.p]},
                              {'params': [L_b for L_b in model.L2] + [W_a_ for W_a_ in model.W_a] + [the for the in
                                                                                                     model.theta],
                               'lr': learning_rate_decoder}],
                             lr=learning_rate, weight_decay=0.001, betas=(0.9, 0.9))
        if epoch_i > 100 and epoch_i % 50 == 0:
            learning_rate = learning_rate / 1.5
            learning_rate_decoder = learning_rate_decoder / 1.5
            opt = optim.Adam([{'params': [L_a for L_a in model.L] + [p for p in model.p]},
                              {'params': [L_b for L_b in model.L2] + [W_a_ for W_a_ in model.W_a] + [the for the in model.theta],
                               'lr': learning_rate_decoder}],
                             lr=learning_rate, weight_decay=0.001, betas=(0.9, 0.9))
        for data_batch in trainloader:
            batch_x, batch_label = data_batch
            output_end, output_abun= model(batch_x.T,A0,batch_label)
            loss=sum([criterion(output_end[i+1] @ output_abun[i+1], batch_x.T) for i in range(layerNum)])/layerNum
            opt.zero_grad()
            loss.backward()
            opt.step()
        for i in range(layerNum):
            t1 = model.p[i].data
            t1[t1 < 0] = 1e-4
            t1[t1 > 1] = 1
            model.p[i].data.copy_(t1)
            running_loss += loss.item()
        temp = abs(running_loss - last_loss) / last_loss
        output_data = 'train===epoch: %d, loss:  %.5f, tol: %.6f\n' % (epoch_i, running_loss, temp)
        print(output_data)
        last_loss=running_loss
        running_loss = 0.0
        if epoch_i % 5 == 0:
            torch.save(model, "%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
    util = UnmixingUtils(A, s.T)
    out1, out2 = model(torch.FloatTensor(X), A0, S0.T)
    Distance, meanDistance, sor = util.hyperSAD(out1[-1].detach().numpy())
    rmse = util.hyperRMSE(out2[-1].T.detach().numpy(), sor)
    output_data = 'Res: SAD: %.5f RMSE:  %.5f' % (meanDistance, rmse)
    print(output_data)
    return meanDistance,rmse
if __name__ == '__main__':
    dataFile ='data/SNR/syntheticDataNewSNR25dB20170601.mat'
    trainFile = 'data/train_4096_500.mat'
    X, A, s = prepare_data(dataFile)
    A0 = np.zeros_like(A)
    S0 = np.zeros_like(s.T)
    for i in range(10):
        initFile = 'data/SNR/syntheticDataNewSNR25dB20170601VCAiter' \
                   + str(i + 1) + 'init.mat'
        _A0, _S0 = prepare_init(initFile)
        util = UnmixingUtils(A, s)
        Distance, meanDistance, sor = util.hyperSAD(_A0)
        rmse = util.hyperRMSE(_S0.T, sor)
        output_data = "L21: SAD ", str(meanDistance), "RMSE: ", str(rmse)
        A0 += _A0[:, sor]
        S0 += _S0[sor, :]
        print(output_data)
    A0 = A0 / 10
    S0 = S0 / 10
    train_data, train_labels, nrtrain = prepare_train(X, S0, trainFile)
    layerNum =9
    lr = 2
    lrD = 1e-6
    # For SNR=15
    # lr = 0.3
    # lrD = 1e-8
    train(lrD=lrD, lr=lr, layerNum=layerNum, train_data=train_data, test_data=train_labels, nrtrain=nrtrain, A0=A0, S0=S0,
          X=X, A=A, s=s.T, SNR='25dB')
