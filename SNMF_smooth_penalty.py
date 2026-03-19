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
from data import HSI
from utils import Metrics

class L1NMF_Net(nn.Module):
    def __init__(self, layerNum, M, A):
        super(L1NMF_Net, self).__init__()
        # abundance step size
        R = np.size(M, 1)
        eig, _ = np.linalg.eig(M.T @ M)
        eig += 0.1
        L = 1 / np.max(eig)
        theta = np.ones((1, R)) * 0.01 * L
        # Endmember step size
        eig, _ = np.linalg.eig(A @ A.T)
        eig += 0.1
        L2 = np.max(eig)
        L2 = 1 / L2

        '''
        Create a smooth penalty matrix D
        '''
        B, N = M.shape
        D = np.eye(B-1,B) - np.eye(B-1,B,k=1)
        # D[104,:] = 0 # Jasper Ridge
        # D[145, :] = 0 # Jasper Ridge

        self.D = nn.ParameterList() # smooth penalty
        self.Ls = nn.ParameterList() # smooth regularization parameter
        self.p = nn.ParameterList()
        self.L = nn.ParameterList()
        self.theta = nn.ParameterList()
        self.L2 = nn.ParameterList()
        self.W_a = nn.ParameterList()
        self.layerNum = layerNum
        temp = self.calW(M)
        for k in range(self.layerNum):
            self.D.append(nn.Parameter(torch.tensor(D)))
            self.Ls.append(nn.Parameter(torch.tensor([3*1e-2])))
            self.L.append(nn.Parameter(torch.tensor([L])))
            self.L2.append(nn.Parameter(torch.tensor([L2])))
            self.theta.append(nn.Parameter(torch.tensor(theta)))
            self.p.append(nn.Parameter(torch.tensor([0.5])))
            self.W_a.append(nn.Parameter(torch.tensor(temp)))
        self.layerNum = layerNum
    def forward(self, X, _M, _A):
        self.W_m = torch.tensor(_A)
        M = list()
        M.append(torch.tensor(_M))
        A = list()
        A.append(torch.tensor(_A.T))
        for k in range(self.layerNum):
            # End-net
            T = M[-1].mm(A[-1]) - X
            T1 = self.D[k].mm(M[-1])
            _M = M[-1] - T.mm(self.L2[k] * A[-1].T) - self.D[k].T.mm(self.L2[k] * self.Ls[k] * T1) # With Smooth penalty
            _M = F.relu(_M)
            M.append(_M)
            # Abun-net
            theta = self.theta[k].repeat(A[-1].size(1), 1).T
            T = M[-1].mm(A[-1]) - X
            _A = A[-1] - self.L[k]*self.W_a[k].T.mm(T)
            _A = self.sum2one(F.relu(self.self_active(_A, self.p[k], theta)))
            A.append(_A)
        return M, A
    def half_thresholding(self, z_hat, mu):
        # l_{1/2} thresholding
        c=pow(54,1/3)/4
        tau=z_hat.abs()-c*pow(mu,2/3)
        v=z_hat
        ind=tau>0
        v[ind]=2/3*z_hat[ind]*(1+torch.cos(2*math.pi/3-2/3*torch.acos(mu[ind]/8*pow(z_hat[ind].abs()/3,-1.5))))
        v[tau<0]=0
        return v
    def soft_thresholding(self, z_hat, mu):
        # l_{1} thresholding
        return z_hat.sign() * F.relu(z_hat.abs() - mu)
    def self_active(self, x, p, lam):
        # Generalized shrinkage Thresholding 0 < p < 1
        tau=pow(2*(1-p)*lam,1/(2-p))+p*lam*pow(2*lam*(1-p), (p-1)/(2-p))
        v = x
        ind = (x-tau) > 0
        ind2=(x-tau)<=0
        v[ind]=x[ind].sign() * (x[ind].abs() - p * lam[ind] * pow(x[ind].abs(), p - 1))
        v[ind2]=0
        v[v>1]=1
        return v
    def calW(self,D):
        # Generalized Mutual Coherence
        (m,n)=D.shape
        W = cp.Variable(shape=(m, n))
        obj = cp.Minimize(cp.norm(W.T @ D, 'fro'))
        # Create one constraint.
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

def prepare_train(X, s, trainFile, size, data=True, seed=0):
    if data:
        train_index = scio.loadmat(trainFile)
        train_index = train_index['train']
        train_index=train_index-1
        # print(train_index)
    else:
        N = X.shape[1]
        rng = np.random.default_rng(seed=seed)
        train_index = rng.choice(np.arange(N), size=size, replace=False)
        train_index = train_index.reshape(-1,1)
        # print(train_index)
    train_data = np.squeeze(X[:, train_index])
    train_labels = np.squeeze(s[:, train_index])
    if data:
        nrtrain = np.size(train_index, 1)
    else:
        nrtrain = size
    return train_data, train_labels, nrtrain

def save_output(Y, A, S, end_init, abun_init, end_est, abun_est, sads, file_ex, rmse):
    my_dict = {"Data":Y, "End_true":A, "Abun_true": S,
               "End_init": end_init, "Abun_init": abun_init,
               "End_est": end_est, "Abun_est": abun_est, "SADs":sads, "RMSE": rmse}
    fileName = file_ex + ".mat"
    scio.savemat(fileName, my_dict)
    return

def save_metrics(SAD, RMSE, x, mName):
    my = {"SAD":SAD, "RMSE": RMSE, "lr_val": x}
    fileName = mName + ".mat"
    scio.savemat(fileName, my)
    return

def prepare_init(initFile):
    init = scio.loadmat(initFile)
    A0 = init['Cn']
    S0 = init['o'][0, 0]['S']
    return A0, S0

def normalize(A):
    return (A - np.min(A))/(np.max(A) - np.min(A))

def SAD_cost(Y_pr, Y):
    Y_tr = torch.tensor(Y)
    norm_tr = torch.linalg.norm(Y_tr, ord=2, axis=0, keepdim=True)
    norm_pr = torch.linalg.norm(Y_pr, ord=2, axis=0, keepdim=True)
    norm = norm_tr * norm_pr
    sad = torch.sum(torch.arccos(torch.diag(Y_tr.T.mm(Y_pr),0)/norm))
    return sad

def SID_cost(Y_pr, Y):
    Y_tr = torch.tensor(Y)
    P = Y_tr/torch.sum(Y_tr, axis=0, keepdim=True)
    Q = Y_pr/torch.sum(Y_pr, axis=0, keepdim=True)
    sid = torch.mean(torch.sum(P * torch.log10(P/Q + 0.01), axis=0) + torch.sum(Q * torch.log10(Q/P + 0.01), axis=0))
    return sid


def set_param(layerNum, lr, lrD, batch_size=4096):
    parser = argparse.ArgumentParser(description="LISTA-Net")
    parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
    parser.add_argument('--end_epoch', type=int, default=1000, help='epoch number of end training')
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
    # batch_size = nrtrain
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
    opt = optim.Adam([{'params': [L_a for L_a in model.L] + [p for p in model.p] + [L_c for L_c in model.Ls]},
                      {'params': [L_b for L_b in model.L2] + [W_a_ for W_a_ in model.W_a] + [the for the in
                                                                       model.theta],
                       'lr': learning_rate_decoder}],
                     lr=learning_rate, weight_decay=0.001, betas=(0.9, 0.9))
    start_epoch = args.start_epoch
    end_epoch = args.end_epoch
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    last_loss = 1
    running_loss = 0

    # Early stopping
    early_stop = 0
    best_loss = np.inf
    rep = 0
    m = 'model'
    for epoch_i in range(start_epoch + 1, end_epoch + 1):
        # print(f"Epoch# {epoch_i}")
        if epoch_i <= 5 and epoch_i % 2 == 0:
            learning_rate = learning_rate / 25
            opt = optim.Adam([{'params': [L_a for L_a in model.L] + [p for p in model.p] + [L_c for L_c in model.Ls]},
                              {'params': [L_b for L_b in model.L2] + [W_a_ for W_a_ in model.W_a] + [the for the in
                                                                                                     model.theta],
                               'lr': learning_rate_decoder}],
                             lr=learning_rate, weight_decay=0.001, betas=(0.9, 0.9))
        if epoch_i > 100 and epoch_i % 50 == 0:
            learning_rate = learning_rate / 1.5
            learning_rate_decoder = learning_rate_decoder / 1.5
            opt = optim.Adam([{'params': [L_a for L_a in model.L] + [p for p in model.p] + [L_c for L_c in model.Ls]},
                              {'params': [L_b for L_b in model.L2] + [W_a_ for W_a_ in model.W_a] + [the for the in model.theta],
                               'lr': learning_rate_decoder}],
                             lr=learning_rate, weight_decay=0.001, betas=(0.9, 0.9))
        for data_batch in trainloader:
            batch_x, batch_label = data_batch
            # print(f"training size: {batch_x.shape},     s0 shape: {batch_label.shape},      Epoch# is {epoch_i}")
            output_end, output_abun= model(batch_x.T,A0,batch_label)
            loss=sum([criterion(output_end[i+1] @ output_abun[i+1], batch_x.T) for i in range(layerNum)])/layerNum
            # print(f"Loss: {loss},   Epoch# number {epoch_i},    Running loss: {running_loss}")
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
        if epoch_i % 100 == 0:
            output_data = 'train===epoch: %d, loss:  %.5f, best loss:  %.5f, tol: %.6f, learning rate: %.7f, learning rate decoder: %.7f\n' % (epoch_i, running_loss, best_loss, temp, learning_rate, learning_rate_decoder)
            print(output_data)
        if running_loss < best_loss and temp < 1e-5:
            best_loss = running_loss
            m = "test_model"
            if not os.path.exists(m):
                os.makedirs(m)
            path = f"{m}/best_model.pkl"
            torch.save(model.state_dict(), path) # save only parameters
        # if epoch_i % 10 == 0:
            # print(m)
            # print(f"{m}/net_params_{epoch_i}.pkl")
            # if epoch_i % 1000 == 0 and (running_loss >= best_loss or temp > 1e-6):
            # break
        # if running_loss <= best_loss:
            # rep += 1

        # if running_loss < best_loss:
        #     rep += 1
        # elif rep >= 100 and temp < 1e-4:
        #     # print(rep) # Debugging
        #     learning_rate_decoder *= 1.1
        #     learning_rate *= 1.1
        #     rep = 0
        '''
        Early stopping
        '''
        # if running_loss >= best_loss and epoch_i > 500 and temp < 1e-5:
        #     early_stop += 1
        #     # print(f"Early Stopping threshold: {early_stop}, Epoch: {epoch_i}") # Debuggin
        #     if early_stop > 20:
        #         break
        # else:
        #     early_stop = 0
        last_loss=running_loss
        running_loss = 0.0
        # if epoch_i % 5 == 0:
        #     torch.save(model, "%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
    util = UnmixingUtils(A, s.T)
    met = Metrics(A, s)
    # Load best model
    if os.path.exists(m):
        path = f"{m}/best_model.pkl"
        # print(m)        
        model.load_state_dict(torch.load(f"{m}/best_model.pkl", weights_only=True))
    out1, out2 = model(torch.FloatTensor(X), A0, S0.T)
    Distance, meanDistance, sor = util.hyperSAD(normalize(out1[-1].detach().numpy()))
    rmse = util.hyperRMSE(out2[-1].T.detach().numpy(), sor)
    dist, meanDist, sor = met.SAD(normalize(out1[-1].detach().numpy()))
    end_rmse, mrmse1 = met.RMSE(out2[-1].detach().numpy(),sor)
    mean_val = meanDistance.item() if hasattr(meanDistance, 'item') else meanDistance
    output_data1 = 'Res: SAD: %.5f RMSE:  %.5f' % (mean_val, rmse)
    print(output_data1)
    output_data = 'Res: SAD: %.5f RMSEL  %.5f' % (meanDist.item(), mrmse1)
    print(output_data)
    return mean_val, rmse
    # return out1[-1].detach().numpy(), out2[-1].detach().numpy(), dist, end_rmse

if __name__ == '__main__':
    # dataFile = "JasperRidge.mat"
    # hsi = HSI(dataFile)
    # dataFile = 'mat_data/Urban_dataFile.mat'
    # data = scio.loadmat(dataFile)
    # data = scio.loadmat(dataFile)
    # X, A, s = hsi()
    # print(f"X max: {X.max()}")
    # print(f"X min: {X.min()}")
    # X = data['Y']
    # A = data['M']
    # s = data['A']
    # parm = X.max()
    # X = X/X.max()
    # H = hsi.H
    # W = hsi.W
    # initFile = "mat_data/JasperRidge_init.mat"
    # data = scio.loadmat(initFile)
    # A0 = data['A0']
    # S0 = data['S0']
    # util = UnmixingUtils(A, s.T)
    # Distance, meanDistance, sor = util.hyperSAD(normalize(A0))
    # rmse = util.hyperRMSE(S0.T, sor)
    # output_data = 'Res: SAD: %.5f RMSE:  %.5f' % (meanDistance.item(), rmse)
    # print(output_data)
    # met = Metrics(A,s)
    # dist, meanDist, sor = met.SAD(normalize(A0))
    # end_rmse, mrmse1 = met.RMSE(S0,sor)
    # output_data = 'Res: SAD: %.5f RMSEL  %.5f' % (meanDist.item(), mrmse1)
    # print(output_data)
    # r = np.random.choice(500,10, replace=False)
    # # list_lrD = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    # # list_lr = [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    # # # list_layerNum = [i for i in range(1,21)]
    # # SAD = []
    # # RMSE = []
    # # print(s.shape)
    # # sees = [12, 30, 42, 46, 62, 67, 50, 78, 88, 92, 100, 121]
    # for i in range(len(sees)):
    #     util = UnmixingUtils(A,s)
    #     mod1 = VCA()
    #     mod2 = FCLSU()
    #     _A0 = mod1.init_like(hsi, seed=sees[i])
    #     _S0 = mod2.solve(X, _A0)
    #     Distance, meanDistnace, sor = util.hyperSAD(_A0)
    #     rmse = util.hyperRMSE(_S0.T, sor)
    #     outputData = "L21: SAD", str(meanDistnace), "RMSE: ", str(rmse)
    #     A0 += _A0[:, sor]
    #     S0 += _S0[sor,:]
    #     print(outputData)
    # A0 = A0 / len(sees)
    # S0 = S0 / len(sees)
    # trainFile = "mat_data/urban_train.mat"
    # k = 1
    # train_data, train_labels, nrtrain = prepare_train(X, S0, trainFile, 500, data=False)
    # layerNum = 12
    # # lr = 0.00001 # For Jasper Ridge
    # # lrD = 1e-8 # For Jasper Ridge
    # # lr = 1e-4 # For Urban
    # # lrD = 1e-8 # For Urban
    # lr = 1e-4 # For Samson
    # lrD = 1e-8 # For Samson
    # # bestlr = -1
    # # bestlrD = -1
    # # bestmd = np.inf
    # # bestrmse = np.inf
    # # end_est, abun_est, sads = train(lrD=lrD, lr=lr, layerNum=layerNum, train_data=train_data, test_data=train_labels, nrtrain=nrtrain, A0=A0, S0=S0,
    # #     X=X, A=A, s=s,SNR='30dB')
    # # for lr in list_lr:
    # #     for lrD in list_lrD:
    # print(f'LayerNum = {layerNum}, Learning Rate = {lr} \n lrD = {lrD}')
    # end_est, abun_est, sads, rmse = train(lrD=lrD, lr=lr, layerNum=layerNum, train_data=train_data, test_data=train_labels, nrtrain=nrtrain, A0=A0, S0=S0,
    # X=X, A=A, s=s,SNR='30dB')
    #     # if mean_dist < bestmd and rmse < bestrmse:
    #     #     bestlrD = lrD
    #     #     bestlr = lr
    #     #     bestmd = mean_dist
    #     #     bestrmse = rmse
    #             # print(bestmd)
    #     # SAD.append(mean_dist)
    #     # RMSE.append(rmse)
    #     # itr += 1
    # # met_name = 'metrics/Samson_lrD'
    # # output = f'data: {dataFile} \nbest overall learning rate: {bestlr} \nbest subset lr {bestlrD}'
    # # print(output)
    # # save_metrics(SAD, RMSE, list_lrD, met_name)
    # # est_name = 'graphs/real_data/JR_seed_' + str(k)
    # est_name = 'Smooth_penalty/JR'
    # save_output(X, A, s.T, A0, S0, end_est, abun_est, sads, est_name, rmse)
    # # k += 1
    # # dataFile ='data/SNR/syntheticDataNewSNR25dB20170601.mat'
    # # trainFile = 'data/train_4096_500.mat'
    # # X, A, s = prepare_data(dataFile)
    # A0 = np.zeros_like(A)
    # S0 = np.zeros_like(s.T)
    # for i in range(10):
    #     initFile = 'data/SNR/syntheticDataNewSNR25dB20170601VCAiter' \
    #                + str(i + 1) + 'init.mat'
    #     _A0, _S0 = prepare_init(initFile)
    #     util = UnmixingUtils(A, s)
    #     Distance, meanDistance, sor = util.hyperSAD(_A0)
    #     rmse = util.hyperRMSE(_S0.T, sor)
    #     output_data = "L21: SAD ", str(meanDistance), "RMSE: ", str(rmse)
    #     A0 += _A0[:, sor]
    #     S0 += _S0[sor, :]
    #     print(output_data)
    # A0 = A0 / 10
    # S0 = S0 / 10
    # train_data, train_labels, nrtrain = prepare_train(X, S0, trainFile)
    # layerNum =9
    # lr = 2
    # lrD = 1e-6
    # # For SNR=15
    # # lr = 0.3
    # # lrD = 1e-8
    # train(lrD=lrD, lr=lr, layerNum=layerNum, train_data=train_data, test_data=train_labels, nrtrain=nrtrain, A0=A0, S0=S0,
    #       X=X, A=A, s=s.T, SNR='25dB')
    SAD = []
    RMSE = []
    ln_list = list(range(3,32,3))
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
    train_data, train_labels, nrtrain = prepare_train(X, S0, trainFile, 500, data=True)
    layerNum =9
    lr = 2
    lrD = 1e-6
    # For SNR=15
    # lr = 0.3
    # lrD = 1e-8
    for layerNum in ln_list:
        sad, rmse = train(lrD=lrD, lr=lr, layerNum=layerNum, train_data=train_data, test_data=train_labels, nrtrain=nrtrain, A0=A0, S0=S0,
            X=X, A=A, s=s.T, SNR='30dB')
        SAD.append(sad)
        RMSE.append(rmse)
    f = 'Smooth_penalty/metrics/layerNum'
    save_metrics(SAD, RMSE, ln_list, f)