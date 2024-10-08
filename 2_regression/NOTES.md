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

### 2.5 - 2.7 Linear Regression
* **Goal**: Given car data $(x_1, \ldots, x_n)$, find a function $g(x_i)$ predict the car price $\hat{y_i}$ for car $x_i$ that is close to actual car price $y$.
* For a 1-dimensional dataset $X$ and vector $y$, linear regression: $\displaystyle y = w_0 + \sum_i^n w_i x_i$, where $w_0$ is the bias term, and $w_1, \ldots, w_n$ are the weights for each feature term.
* To simplify the equation, prepend X (features) with all-ones vector to include the bias terms, so we could solve for $w$ using $y=X\cdot w$, where $w = w_0, w_1,\ldots, w_n$.
* The solution is $w = X^{-1}y$, however this implies an assumption that $X$ is a square matrix, which in most cases it is not.
* To ensure that we could find the inverse matrix, we need to ensure that the matrix itself is a square matrix.
* Hence we first find the square matrix using Gram Matrix $X^T X$, then solve the following:
  $w = (X^T X)^{-1} X^T y$.
* The solution $w$ is then split into weights (biases removed) and biases (first column of $w$).

### 2.8 Baseline model
* The baseline model is defined using only features engine_hp, engine_cylinders, highway_mpg, city_mpg, popularity.
* A subset of training data with only these features is used as training data.
* Missing values in the subset are set to 0.
* Training is done using linear regression from previous section 2.5 - 2.7.
* The weights and biases are then used to produce a set of predictions with the subset data.
* The predictions and actual car prices are then compared in a histogram chart.

### 2.9 - 2.10 Root Mean Squared Error RMSE
* Goal: To measure the std deviation of the errors (difference between predictions and actual values).
* Formula: $\displaystyle \text{RMSE} = \sqrt{\frac{1}{m}\sum_{i=1}^m(y_i - g(x_i))^2}$, $m$: total records in dataset.
* RMSE is then applied onto training and validation dataset.

### 2.11 Feature Engineering
* Includes car age as a new feature for the subset data.
* Train model and evaluate performance with RMSE and histogram chart.

### 2.12 Categorical Variables
* Discrete values in data usually and a limited set of unique values, typically alphanumeric but numeric is also common.

### 2.13 - 2.14 Regularization and tuning the model
* An inverse of the Gram matrix may not exist due to duplicate columns, i.e. non linearly-independent columns within the matrix. These are usually known as singular matrices, i.e. matrix without inverse matrix.
* Due to noise in dataset, it may be possible to have columns that are very similar in values but would not throw off the singular matrix error.
* To mitigate this, a regularization factor $\alpha\mathbb{I}$ is added to the Gram Matrix.
* $w = (X^T X + \lambda\mathbb{I})^{-1} X^T y$
* Further reading: [Regularization in Regression](https://datatalks.club/blog/regularization-in-regression.html)
* A range of different regularization factors are experimented on and a final factor is then decided based on the resulting RMSE.

### 2.15 Using the model
* Predict using a test record.
* Convert car price from log form to usual form by taking exponents.

### 2.16 Summary
* We prepare our data and understand it better using EDA.
* We split our data into training, validation and test sets.
* We use a baseline model to train a linear regression model.
* We evaluate model performance using RMSE.
* We try to improve the model using feature engineering and regularization.
* We evaluate again the model performance and decided on the final model.
* We predict car price from test data using the final model.

### Homework
* Module 2 [Homework questions](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/cohorts/2024/02-regression/homework.md)