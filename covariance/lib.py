
from abc import abstractmethod
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

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

class Hemisphere:
    North = 'North'
    South = 'South'

    @staticmethod
    def latitude(lat: float) -> 'Hemisphere':
        return Hemisphere.North if lat >= 0 else Hemisphere.South

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
            for _, point in tqdm(test_set.iterrows(), total=len(test_set))
        ])

    @classmethod
    def assess(cls, train_set: pd.DataFrame, test_set: pd.DataFrame) -> float:
        """
        Trains the model on the training set, then computes and pretty prints some test set statistics
        You can turn it into the implied geometric mean probability density by taking math.exp(return_value)
        """
        self = cls()
        name = self.__class__.__name__

        self.train(train_set)

        # log likelihood of the entire test set, usually very negative (eg -100_000)
        log_likelihood = self.log_likelihood(test_set)
        # ln (geometric mean of likelihoods on the test set), usually a better stat for comparison
        # This is a useful metric for comparing models: the less negative is better
        log_geo_mean_likelihood = log_likelihood / len(test_set)
        # geometric mean of likelihoods on the test set, usually a relatively small number
        geo_mean_p_density = math.exp(log_geo_mean_likelihood)

        print(
            f"{name}:"
            f"\n  log likelihood: {log_likelihood:.0f}"
            f"\n  log geo mean likelihood: {log_geo_mean_likelihood:.3f}"
            f"\n  geo mean p density: {geo_mean_p_density:.5f}"
        )
