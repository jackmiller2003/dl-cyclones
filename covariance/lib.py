
import math
import numpy as np

def log_zeromean_multivariate_normal_pdf(x, cov): # x is a vector, cov is a square covariance matrix
    k = cov.shape[0]
    return -0.5 * (k * math.log(2*math.pi) + math.log(np.linalg.det(cov)) + x.T @ np.linalg.pinv(cov) @ x)
