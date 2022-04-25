
# pip install xgboost tensorflow scikit-learn

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

    lon0, lat0 = y_true[:,0], y_true[:,1]
    lon1, lat1 = y_pred[:,0], y_pred[:,1]

    phi0 = lat0 * (math.pi/180) # radians
    phi1 = lat1 * (math.pi/180)

    delta_phi = phi1 - phi0
    delta_lambda = (lon1 - lon0) * (math.pi/180)

    a = np.sin(delta_phi/2)**2 + np.cos(phi0) * np.cos(phi1) * np.sin(delta_lambda/2)**2
    c = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return c

def haversine_loss_tf(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    R = 6371 # km

    lon0, lat0 = y_true[:,0], y_true[:,1]
    lon1, lat1 = y_pred[:,0], y_pred[:,1]

    phi0 = lat0 * (math.pi/180) # radians
    phi1 = lat1 * (math.pi/180)

    delta_phi = phi1 - phi0
    delta_lambda = (lon1 - lon0) * (math.pi/180)

    a = tf.math.sin(delta_phi/2)**2 + tf.math.cos(phi0) * tf.math.cos(phi1) * tf.math.sin(delta_lambda/2)**2
    c = 2 * R * tf.math.atan2(np.sqrt(a), tf.math.sqrt(1-a))

    return c

def load_model(name, alias):
    try:
        model = pickle.load(open(f'models/pkl/{name}-{alias}.pkl', 'rb'))
        if name == 'ANN': model.src = keras.models.load_model(f'models/pkl/{name}-{alias}.h5')
        return model
    except Exception as e:
        print_exc()

class BaseModel:
    def __init__(self, name):
        self.name = name
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
        pickle.dump(self, open(f'models/pkl/{self.name}-{alias}.pkl', 'wb'))


# ARTIFICIAL NEURAL NETWORK

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

class ANNModel(BaseModel):
    def __init__(self):
        super().__init__('ANN')

    def train(self, Xt, Xv, Yt, Yv, verbose=False):
        model = Sequential([
            Dense(256, input_shape=Xt.shape[1:], activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(256, input_shape=(256,), activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(2)
        ])

        if verbose: model.summary()

        model.compile(
            loss=haversine_loss_tf,
            optimizer='adam',
            metrics=[]
        )

        batch = 100
        history = model.fit(
            Xt, Yt,
            batch_size=batch,
            epochs=10,
            steps_per_epoch=(Xt.shape[0] // batch),
            verbose=1,
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
        self.src.save(f'models/pkl/{self.name}-{alias}.h5')
        model = self.src
        self.src = None
        super().save(alias)
        self.src = model


# ADAPTIVE BOOSTING (ADABOOST)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

class AdaBoostModel(BaseModel):
    def __init__(self):
        super().__init__('ADA')

    def train(self, Xt, Xv, Yt, Yv, verbose=False):
        models = GridSearchCV(AdaBoostClassifier(), { "n_estimators": [30, 300] }, verbose=verbose)
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


# K NEAREST NEIGHBOURS (K-NN)

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

class KNNModel(BaseModel):
    def __init__(self):
        super().__init__('KNN')

    def train(self, Xt, Xv, Yt, Yv, verbose=False):
        params = {
            "n_neighbors": [2, 5, 10],
            "weights": ["uniform"],
            "metric": ["manhattan"]
        }

        models = GridSearchCV(KNeighborsClassifier(n_jobs=-1), params, verbose=verbose)
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

from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__('RF')

    def train(self, Xt, Xv, Yt, Yv, verbose=False):
        params = {
            "max_depth": [None],
            "max_features": [200],
            "n_estimators": [30]
        }

        models = GridSearchCV(RandomForestClassifier(verbose=verbose), params, verbose=verbose)
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

from xgboost import XGBClassifier

class XGBModel(BaseModel):
    def __init__(self):
        super().__init__('XGB')

    def train(self, Xt, Xv, Yt, Yv, verbose=False):
        model = XGBClassifier(verbose=verbose)
        model.fit(Xt, Yt)
        if verbose: print('fit model')
        self.src = model
        self.mean_km = haversine_loss(Yv, model.predict(Xv)).mean()
        if verbose: print(f'validation mean km error: {self.mean_km:.1f}')

    def predict(self, X):
        return self.onehot_from_cat(self.src.predict(X))


# ENSEMBLE

model_classes = [ANNModel, AdaBoostModel, KNNModel, RandomForestModel, XGBModel]

# each row is the feature vector for a time interval
train_feature_vectors = np.load('train_features.npy')
valid_feature_vectors = np.load('valid_features.npy')

# each row is the label vector (tuple of long, lat displacements) for a time interval
train_label_vectors = np.load('train_labels.npy')
valid_label_vectors = np.load('valid_labels.npy')

# train all of the models in model_classes on the training data
for model_class in model_classes:
    model = model_class()
    print("Training model " + model.name)
    model.train(train_feature_vectors, train_label_vectors, valid_feature_vectors, valid_label_vectors, verbose=True)
    print("Saving model " + model.name + " which had validation mean km error of " + str(model.mean_km))
    model.save(model.name)
