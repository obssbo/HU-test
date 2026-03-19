import numpy as np
import scipy.io as scio
import torch
from utils import Metrics
from UnmixingUtils import UnmixingUtils
from data import HSI

class L_halfNMF:
    def __init__(self, X, p):
        """
        :param X: Hyperspectral image of shape LxN where L is number of bands
                   and N number of pixels
        :param p: Number of endmembers
        """
        self.p = p
        self.X = X

    def algorithm(self, delta):
        '''
        Input
        :param delta: regularizing parameter for softly enforcing ASC

        Output
        :param A_f: Estimated Endmembers from L_{1/2} NMF algorithm
        :param S: Estimated Abundances from L_{1/2} NMF algorithm
        '''
        L, N = self.X.shape
        A = np.random.uniform(0,1,(L,self.p))
        S = np.random.uniform(0,1,(self.p,N))
        S = S / np.sum(S, axis=0)

        # Sparsity parameter
        lam = self.get_lam(L,N)
        print(lam)
        C_old = 0
        C_new = np.inf

        if delta > 0:
            # Softly enforce ASC
            one_X = np.ones((N,1))
            one_A = np.ones((self.p,1))
            X_f = np.vstack((X, delta * one_X.T))
            A_f = np.vstack((A, delta * one_A.T))
        else:
            X_f = X.copy()
            A_f = A.copy()
        i = 1
        while (np.abs(C_old - C_new) > 1e-4) and (i < 10000):            
            C_old = np.linalg.norm(X_f - A_f @ S, ord='fro')**2
            # Update A according to L_{1/2} NMF eq. 11
            A_f = A_f * (X_f @ S.T) / (A_f @ S @ S.T)
            if delta > 0:
                A_f[-1,:] = delta * 1
            # Update S according to L_{1/2} NMF eq. 12
            S = S * (A_f.T @ X_f) / (A_f.T @ (A_f @ S) + lam / 2 * np.sum(np.sqrt(S)))
            C_new = np.linalg.norm(X_f - A_f @ S, ord='fro')**2
            i += 1
            if i % 500 == 0:
                print(f"Iteration number: {i}, - Tolerance: {np.abs(C_old - C_new)}")
        return X_f, A_f[:L,:], S


    def get_lam(self,H,W):
        x1 = np.linalg.norm(self.X.T,ord=1, axis=0)
        x2 = np.linalg.norm(self.X.T, ord=2, axis=0)
        lam = np.sum((np.sqrt(W) - x1/x2)/(np.sqrt(W-1)))/np.sqrt(H)
        return lam
    
def save_out(end, abun, fileName):
    my_dict = {"A0": end, "S0":abun}
    scio.savemat(fileName, my_dict)

def prepare_data(dataFile):
    data = scio.loadmat(dataFile)
    X = data['x_n']
    A = data['A']
    s = data['s']
    return X, A, s

if __name__ == "__main__":
    # f = 'JasperRidge.mat'
    # hsi = HSI(f)
    # X, A, s = hsi()
    snr_list = ['15dB', '20dB', '25dB', '30dB', 'Inf']
    # for snr in snr_list:
    snr = snr_list[3]
    f = 'data/SNR/syntheticDataNewSNR' + snr + '20170601.mat'
    X, A, s = prepare_data(f)
    X = np.maximum(0,X)
    A0 = np.zeros_like(A)
    S0 = np.zeros_like(s.T)
    for i in range(10):
        mod = L_halfNMF(X, 6)
        d = 2 # ASC regularization
        _, _A0, _S0 = mod.algorithm(delta=d)
        print(f"ASC parameter delta: {d}")
        u = Metrics(A, s.T)
        Distance, meanDist, sor = u.SAD(A0)
        A0 += _A0[:,sor]
        rmse_list, meanRMSE = u.RMSE(S0, sor)
        S0 += _S0[sor,:]
        output_data = 'Res: SAD: %.5f RMSE:  %.5f' % (meanDist, meanRMSE)
        print(output_data)
    A0 = A0 / 10
    S0 = S0 / 10
    Distance, meanDist, sor = u.SAD(A0)
    rmse_list, meanRMSE = u.RMSE(S0, sor)
    output_data = 'Res: SAD: %.5f RMSE:  %.5f' % (meanDist, meanRMSE)
    print(output_data)

    fileName = 'l_half_nmf/syntheticDataNewSNR' + snr + '_lHalfNMF_d' + str(d) + '.mat'
    save_out(A0,S0,fileName)