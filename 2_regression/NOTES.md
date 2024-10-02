## Regression
[Repo](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/) for module 2 lessons. Module 2 slides refer [here](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides).

### 2.1 Car Price Prediction Project
* [Kaggle dataset source](https://www.kaggle.com/datasets/CooperUnion/cardataset), the notebook and [dataset](https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/refs/heads/master/chapter-02-car-price/data.csv) used for this module are available at [mlbookcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/chapter-02-car-price). 
* Project plan:
    * EDA
    * Use Linear Regression to predict price
    * Understand the internals of linear regression
    * Evaluate using RMSE
    * Feature engineering
    * Regularization
    * Using the model

### 2.2 Data Preparation
* Load car price dataset
* Data preprocessing:Standardize column names, replace whitespace values with underscores.

### 2.3 Exploratory Data Analysis
* Explore the car price distribution.
* Perform log transformation to normalize car prices.

### 2.4 Setting up the validation framework
* Shuffle data and split into training, validation and test datasets.
* Split target variable (car price) from these datasets.
* Log transformation on target variables.

### 2.5 Linear Regression
* **Goal**: Given car data $(x_1, \ldots, x_n)$, find a function $g(x_i)$ predict the car price $\hat{y_i}$ for car $x_i$ that is close to actual car price $y$.
