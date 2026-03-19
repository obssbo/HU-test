import numpy as np
import scipy.linalg as LA
from scipy.optimize import linear_sum_assignment

class Metrics:
    def __init__(self, A, s):
        self.A = A
        self.S = s
    
    def SAD(self, A_est):
        print("A_est has nan:", np.isnan(A_est).any())
        print("A_est has inf:", np.isinf(A_est).any())
        print("A norms:", LA.norm(self.A, axis=0))
        print("A_est norms:", LA.norm(A_est, axis=0))
        L, p = A_est.shape
        # Build cost matrix
        cost = np.zeros((p, p))
        for k in range(p):
            for j in range(p):
                val = np.clip(
                    np.dot(self.A[:, k], A_est[:, j]) /
                    (LA.norm(self.A[:, k]) * LA.norm(A_est[:, j]) + 0.01),
                    -1.0, 1.0
                )
                cost[k, j] = np.arccos(val)
        
        row_ind, col_ind = linear_sum_assignment(cost)
        Dist = cost[row_ind, col_ind]
        return Dist, Dist.mean(), col_ind
    
    def RMSE(self, S_est, idx):
        p, N = S_est.shape
        S = S_est[idx, :]
        diffs = np.sqrt(LA.norm(self.S - S, ord=2, axis=1)**2 / N)
        return diffs.tolist(), diffs.mean()
