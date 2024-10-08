## Classification
[Repo](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/03-classification) for module 3 lessons. Slides available [here](https://www.slideshare.net/slideshow/ml-zoomcamp-3-machine-learning-for-classification/250224470).

### 3.1 Churn prediction project
* Goal: Given a [telco customer dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), predict if the customer would churn, i.e. leave.
* This is done using binary classification, where based on a set of features, the likelihood of customer churn is first predicted then classified into 0 (unlikely to churn) or 1 (likely to churn).
* These predictions help the telco company to perform targeted marketing to customers likely to churn, so as to keep these customers to retain their subscriptions with the company.

### 3.2 Data preparation
* Load dataset, transform churn column from string to integer, transform charges from alphanumeric to numeric form and replace strings with "-" to 0.

### 3.3 Setting up the validation framework
* Split dataset into train, validation, test using sklearn package.

### 3.4 EDA
* Check on missing values, churn statistics and categorical values.
* The **churn rate** can be calculated using mean since churn $\in \{0,1\}$, hence mean $= \displaystyle \frac{1}{n}\sum_i x_i$ only takes the percentage of customers who churned in the dataset.

### 3.5 Feature importance
* Investigate on the churn rate by groups within a category v.s. global churn rate (from 3.4).
* Metrics used: 
    * difference = global - group. Difference < 0 represents higher likelihood to churn.
    * risk ratio = group/global. Risk ratio > 1 represents higher likelihood to churn.
* 
### 3.6 Mutual Information (MI)
* Further reading: [Wikipedia Mutual Information](https://en.wikipedia.org/wiki/Mutual_information)
* In information theory, MI measures the mutual independence between two variables.
* This gives a more uniform metric when we compare the feature importance between feature variables and the churn.
