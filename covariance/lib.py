
from abc import abstractmethod
import math
import numpy as np
import pandas as pd

def log_zeromean_multivariate_normal_pdf(x, cov): # x is a vector, cov is a square covariance matrix
    k = cov.shape[0]
    return -0.5 * (k * math.log(2*math.pi) + math.log(np.linalg.det(cov)) + x.T @ np.linalg.pinv(cov) @ x)

def df_residuals(df: pd.DataFrame):
    pred = np.column_stack((df.pred_long_disp.to_numpy(), df.pred_lat_disp.to_numpy()))
    true = np.column_stack((df.true_long_disp.to_numpy(), df.true_lat_disp.to_numpy()))
    return true - pred

def point_residual(point: pd.Series):
    return np.array([point.true_long_disp - point.pred_long_disp, point.true_lat_disp - point.pred_lat_disp])

def pretty_cov(cov: np.ndarray):
    return f"[[{cov[0][0]:.3f}, {cov[0][1]:.3f}], [{cov[1][0]:.3f}, {cov[1][1]:.3f}]]"

class CovarianceModel:

    """
    This model is designed to compute an estimate of a 2x2 covariance matrix (on longitude+latitude tuples)
    to estimate the error distribution of our cyclone trajectory predictions, assuming a normal distribution

    The core problem is to find a function f(long, lat, intensity) -> covariance matrix
    which maximises the log likelihood of the validation/test set

    We can train a model using the predicted and true displacements of the cyclones in the training set
    """

    @abstractmethod
    def train(self, train_set: pd.DataFrame):
        pass

    @abstractmethod
    # returns a 2x2 covariance matrix
    def estimate(self, long: float, lat: float, intensity: float) -> np.ndarray:
        pass

    def log_likelihood(self, test_set: pd.DataFrame) -> float:
        return sum([
            log_zeromean_multivariate_normal_pdf(
                point_residual(point),
                self.estimate(point.long, point.lat, point.intensity)
            )
            for _, point in test_set.iterrows()
        ])

    def assess_geo_mean_log_likelihood(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> float:
        self.train(train_set)
        return self.log_likelihood(test_set) / len(test_set)
