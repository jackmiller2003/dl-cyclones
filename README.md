# dl-cyclones

This repository has the code we used to predict cyclone trajectories with deep learning models, and to estimate the error distribution of our prediction models.

The `tracks` directory has some scripts and data from the IBTrACS cyclone track database. We process the database into a JSON file with time-series position data for a few thousand cyclones worldwide from 1980 onwards.

The `reanalysis` directory has some scripts and notebooks for extracting and pre-processing data from the ERA-5 reanalysis database. This database has good estimates of atmospheric pressure, wind, etc. data in a fine grid at regular time increments for the last 40 years. We use the copy of this dataset on the [NCI](https://nci.org.au/) and extract pressure and wind in windows centred around the cyclone tracks from IBTrACS.

The `networks` directory has some deep learning models (various convolutional neural networks) and associated infrastructure (training pipelines, data loaders, loss functions, etc.). The models are mostly of the form "predict the displacement of this cyclone over the next few hours given a heatmap of pressure/wind data".

The `covariance` directory has some notebooks where we construct and validate models of the error distribution of our conv-nets. This is the key contribution of our work. We model the error/residuals of our model (how far the cyclone ended up from where we thought) as a 0 mean 2D Gaussian and use various techniques to estimate the covariance matrix of this Gaussian. For more information on how we do this, see [this doc](https://docs.google.com/document/d/1wP4q-afFNYrirfg9QCk-P-SJyDBvk6dNOxfofFxIwpE/edit?usp=sharing).
