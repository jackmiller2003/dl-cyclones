
# pip install xgboost tensorflow sklearn

import pickle
from traceback import print_exc
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import math

"""
These stacked models are trained using the feature vectors from midway through the two CNN's and the metadata concatenated together.
They are used to predict the lat/long angle displacement. The loss function is the mean squared error of the great circle distance
in kilometres calculated using the Haversine formula.
"""

def haversine_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    R = 6371 # km
    
    """
    [[129.09   129.047 ]
      [ 25.9511  28.0132]]
    """
    
    pred_location = y_true[:,0] + y_pred
    true_location = y_true[:,1]
    
    lon0, lat0 = true_location[0], true_location[1]
    lon1, lat1 = pred_location[0], pred_location[1]

    phi0 = lat0 * (math.pi/180) # radians
    phi1 = lat1 * (math.pi/180)

    delta_phi = phi1 - phi0
    delta_lambda = (lon1 - lon0) * (math.pi/180)

    a = np.sin(delta_phi/2)**2 + np.cos(phi0) * np.cos(phi1) * np.sin(delta_lambda/2)**2
    c = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    c = np.sum(c)/y_true.shape[0]

    return c

def haversine_loss_tf(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    R = 6371 # km
    
    pred_location = tf.math.add(y_true[:,:,0], y_pred)
    true_location = y_true[:,:,1]
    
    lon0, lat0 = true_location[:,0], true_location[:,1]
    lon1, lat1 = pred_location[:,0], pred_location[:,1]

    phi0 = lat0 * (math.pi/180) # radians
    phi1 = lat1 * (math.pi/180)

    delta_phi = phi1 - phi0
    delta_lambda = (lon1 - lon0) * (math.pi/180)

    a = tf.math.sin(delta_phi/2)**2 + tf.math.cos(phi0) * tf.math.cos(phi1) * tf.math.sin(delta_lambda/2)**2
    c = 2 * R * tf.math.atan2(tf.math.sqrt(a), tf.math.sqrt(1-a))
    
    return c

class BaseModel:
    NAME = None

    def __init__(self):
        self.mean_km = 0 # validation mean KM error
        self.src = None # underlying model object

    def train(self, Xt, Xv, Yt, Yv, verbose=False):
        # takes in training and validation X and Y arrays
        self.acc = 0
        self.src = None

    def predict(self, X) -> np.ndarray:
        # return an Nx2 array of predictions
        return None

    def save(self, alias):
        pickle.dump(self, open(f'models/pkl/{self.NAME}-{alias}.pkl', 'wb'))

    @classmethod
    def load(cls, alias):
        return pickle.load(open(f'models/pkl/{cls.NAME}-{alias}.pkl', 'rb'))


# ARTIFICIAL NEURAL NETWORK

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LayerNormalization

class ANNModel(BaseModel):
    NAME = 'ANN'

    def train(self, Xt, Xv, Yt, Yv, verbose=False):
        model = Sequential([
            Dense(2048, input_shape=Xt.shape[1:], activation='gelu', kernel_regularizer='l1_l2'),
            BatchNormalization(),
            Dense(Yt.shape[1], input_shape=(2048,), activation='gelu', kernel_regularizer='l1_l2')
        ])

        if verbose: model.summary()

        optimizer = keras.optimizers.Adam(learning_rate=1e-4)
        
        model.compile(
            loss=haversine_loss_tf,
            optimizer=optimizer,
            metrics=[]
        )

        batch = 512
        history = model.fit(
            Xt, Yt,
            batch_size=batch,
            epochs=100,
            steps_per_epoch=(Xt.shape[0] // batch),
            verbose=2,
            shuffle=True,
            validation_data=(Xv,Yv)
        )

        self.src = model
        self.mean_km = haversine_loss(Yv, model.predict(Xv)).mean()
        if verbose: print(f'validation mean km error: {self.mean_km:.1f}')

    def predict(self, X):
        return self.src.predict(X)

    def save(self, alias):
        # save actual model separately in h5 file
        self.src.save(f'models/pkl/{self.NAME}-{alias}.h5')
        model = self.src
        self.src = None
        super().save(alias)
        self.src = model

    @classmethod
    def load(cls, alias):
        model = pickle.load(open(f'models/pkl/{cls.NAME}-{alias}.pkl', 'rb'))
        model.src = keras.models.load_model(f'models/pkl/{cls.NAME}-{alias}.h5',
            custom_objects={ "haversine_loss_tf": haversine_loss_tf })
        return model


# ADAPTIVE BOOSTING (ADABOOST)

from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV

class AdaBoostModel(BaseModel):
    NAME = 'ADA'

    def train(self, Xt, Xv, Yt, Yv, verbose=False):
        output_dim = len(Yt[0])
        # AdaBoostRegressor only works with real targets, but we want to operate on vectors
        # Thus (I kid you not) we train a vector of regression models, one for each component
        models = [GridSearchCV(AdaBoostRegressor(), { "n_estimators": [30] }, verbose=verbose) for i in range(output_dim)]
        for i, model in enumerate(models): model.fit(Xt, Yt[:,i])
        if verbose: print('fit models')

        bests = [model.best_estimator_ for model in models]
        print([model.best_params_ for model in models])
        for i, best in enumerate(bests): best.fit(Xt, Yt[:,i])
        if verbose: print('fit best model')

        self.src = bests
        preds = [best.predict(Xv) for best in bests]
        self.mean_km = haversine_loss(Yv, np.array(preds).T).mean()
        if verbose: print(f'validation mean km error: {self.mean_km:.1f}')

    def predict(self, X):
        return np.array([best.predict(X) for best in self.src]).T


# K NEAREST NEIGHBOURS (K-NN)

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

class KNNModel(BaseModel):
    NAME = 'KNN'

    def train(self, Xt, Xv, Yt, Yv, verbose=False):
        params = {
            "n_neighbors": [10],
            "weights": ["uniform"],
            "metric": ["manhattan"]
        }

        models = GridSearchCV(KNeighborsRegressor(n_jobs=-1), params, verbose=verbose)
        models.fit(Xt, Yt)
        if verbose: print('fit models')

        best = models.best_estimator_
        best.fit(Xt, Yt)
        if verbose: print('fit best model')

        self.src = best
        self.mean_km = haversine_loss(Yv, best.predict(Xv)).mean()
        if verbose: print(f'validation mean km error: {self.mean_km:.1f}')

    def predict(self, X):
        return self.src.predict(X)


# RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor

class RandomForestModel(BaseModel):
    NAME = 'RF'

    def train(self, Xt, Xv, Yt, Yv, verbose=False):
        params = {
            "max_depth": [None],
            #"max_features": [200],
            "n_estimators": [30]
        }

        models = GridSearchCV(RandomForestRegressor(verbose=verbose), params, verbose=2)
        models.fit(Xt, Yt)
        if verbose: print('fit models')

        best = models.best_estimator_
        best.fit(Xt, Yt)
        if verbose: print('fit best model')

        self.src = best
        self.mean_km = haversine_loss(Yv, best.predict(Xv)).mean()
        if verbose: print(f'validation mean km error: {self.mean_km:.1f}')

    def predict(self, X):
        return self.src.predict(X)


# GRADIENT BOOSTING (XGBOOST)

from xgboost import XGBRegressor

class XGBModel(BaseModel):
    NAME = 'XGB'

    def train(self, Xt, Xv, Yt, Yv, verbose=False):
        model = XGBRegressor(verbose=verbose)
        model.fit(Xt, Yt)
        if verbose: print('fit model')
        self.src = model
        self.mean_km = haversine_loss(Yv, model.predict(Xv)).mean()
        if verbose: print(f'validation mean km error: {self.mean_km:.1f}')

    def predict(self, X):
        return self.src.predict(X)


# ENSEMBLE

model_classes = [ANNModel, AdaBoostModel, KNNModel, RandomForestModel, XGBModel]

if __name__ == "__main__":

    # each row is the feature vector for a time interval
    train_feature_vectors = np.load('train_features.npy')
    valid_feature_vectors = np.load('valid_features.npy')

    # each row is the label vector (tuple of long, lat displacements) for a time interval
    train_label_vectors = np.load('train_labels.npy')
    valid_label_vectors = np.load('valid_labels.npy')

    # train all of the models in model_classes on the training data
    for model_class in model_classes:
        model = model_class()
        print("Training model " + model.NAME)
        model.train(train_feature_vectors, train_label_vectors, valid_feature_vectors, valid_label_vectors, verbose=True)
        print("Saving model " + model.NAME + " which had validation mean km error of " + str(model.mean_km))
        model.save('test')
