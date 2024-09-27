## Intro to Machine Learning
Module 1 slides available [here](https://www.slideshare.net/slideshow/ml-zoomcamp-11-introduction-to-machine-learning/250094645). [Link](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/01-intro/01-what-is-ml.md) to module 1 lessons.

### 1.1 Intro to ML
* Used car dealership price prediction example: based on a car's year, make, mileage a dealer could predict the price of a car.
* Machine Learning (ML): Process of extracting patterns from data (both features and target).
* Machine Learning model: Encapsulates the learned patterns from the machine learning process which helps it make predictions on unseen data.
* In model training, features and target of past data are fed into a ML process to produce an ML model which could then be used to predict targets based on features only.

### 1.2 ML vs Rule-based system
* Use case: Spam detection system. Given an email, classify it as spam/not spam.
* Rule-based spam detection system defines a set of concrete rules to filter out spam, e.g. filter out all emails that contain specific subject titles or from a particular sender.
* Cons with rule-based spam detection system: 
    * May filter out legitimate emails which fall under the rules for spam emails.
    * Spams keeps changing so rules keep changing.
    * Difficult to maintain when the number of rules keeps incrementing, until it gets out of hand.
* ML spam detection system learns the characteristics of spam and legitimate emails, e.g. email content, sender address, and user labels (spam/not spam). This helps to train an ML model which helps to predict a label if an email is spam/not spam.
* The prediction could be a probability between 0 and 1, and we could then set a cutoff such as: probability it's a spam $\mathbb{P}(\text{spam}) \ge 0.5$ will result in a target label = "spam".

### 1.3 Supervised ML
* Idea: As per spam detection system example, supervised ML is the process of learning the patterns where there are target labels being provided in the learning process. This helps the ML model to be able to generalise well to unseen examples (new data) to predict what the target labels would be.
* The feature data is represented in a matrix, i.e. feature matrix $X$ and target vector $y$.
* Each row in feature matrix is a data record, each column is a feature.
* What happens is $g(X) \approx y$ where $g(X)$ is a function that accept the feature matrix $X$ to produce a predicted target that is approx. close to the actual target $y$.
* Examples of supervised ML: 
    * Regression: outputs a number as target.
    * Classification: outputs a category as target. There are binary class and multi-class types in this.
    * Ranking: e.g. recommender system. Outputs probabilities as target and returns the top $k$ probabilities as recommendations. 

### 1.4 CRISP-DM (ML process)
* CRoss Industry Standard Process for Data Mining: methodology for organising ML projects.
![crisp-dm diagram](https://upload.wikimedia.org/wikipedia/commons/b/b9/CRISP-DM_Process_Diagram.png)
* Step 1: Business understanding. Identiy the business problem, understand how we can solve it.
    * Important question: Do we actually need ML here?
    * Define a measurable goal.
* Step 2: Data understanding. Identify the data sources.
    * Is data reliable?
    * Do we need more data?
    * Do we track the data correctly?
    * It may influence the goal and we may have to go back to adjust it.
* Step 3: Data preparation. 
    * Feature extraction.
    * Data cleaning.
    * Building pipelines: raw data $\rightarrow$ transformation $\rightarrow$ clean data.
    * Convert into tabular data.
* Step 4: Modeling.
    * Training model.
    * Try a few models and pick the best one.
    * May go back to data preparation.
* Step 5: Evaluation.
    * Evaluate against the goal.
* Evaluation + deployment: Online evaluation using live data (after deployment) .
* Step 6: Deployment.
    * Rollout to all users.
    * Proper monitoring.
    * Best engineering practices to ensure quality and maintainability.
* Iterate! ML projects require many iterations.
    * Start simple.
    * Learn from feedback.
    * Improve.

### 1.5 Modeling step: Model selection
* Holdout + train: Holdout 20% data as validation dataset $X_v$ and use the 80% data $X$ for training.
* Making predictions: From training we get a function $g(X) = y$, and then apply $g(X)$ function onto validation dataset to make predictions, hence $g(X_v) = \hat{y_v}$
* Scoring: We then compare $\hat{y_v}$ and actual targets from holdout data $y_v$ to see how similar the predictions are to the actual targets. 
    * $g(X)$ could be models such as Logistic Regression, Decision Tree, Random Forest, Neural Networks etc. And the comparison produces a score for each model, which helps to decide which model works best for the use case.
* Multiple comparisons problem: A model may be very good at predicting a particular holdout data but doesn't generalise well. 
* Train + Validation + Test: To prevent the problem earlier, we use multiple holdout data, e.g. 60% training data, 20% validation data, 20% test data. We train and then score the models using validation data and pick the best one. 
* Further scoring: Using the test data, we further score the best model on this. If the new score is reasonably close to the score using validation data, we can conclude the best model performs well and not a multiple comparison problem.
* Model selection (6 steps):
    * Split dataset into train, validation, test data.
    * Train models.
    * Validate models.
    * Select the best model.
    * Test the best model using test data.
    * Check scores between validation and test data.

### 1.6 Environment Setup
* See [course notes](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/01-intro/06-environment.md).
* My virtual env:
```bash
conda create -n ml-zoomcamp-env python=3.12
conda activate ml-zoomcamp-env
conda install numpy pandas scikit-learn seaborn jupyter
```
* My AWS env: Refer to `.env`, `*_credentials.csv`, `*accessKeys.csv` within project folder (not available in Github repo).

### 1.7 Introduction to Numpy
* Sample matrix operations with Numpy can be found in [course notes](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/01-intro/07-numpy.md).
* Also see [DataCamp's Numpy Cheat Sheet](https://www.datacamp.com/community/blog/python-numpy-cheat-sheet).

### 1.8 Linear Algebra Refresher
* Linear Algebra course notes, sample notebook available [here](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/01-intro/08-linear-algebra.md).
* Vector operations: element-wise operations
* Multiplication
    * Vector-vector multiplication
    * Matrix-vector multiplication
    * Matrix-matrix multiplication
* Identity matrix
* Inverse: The inverse matrix of $X$ is $X^{-1}$ where $X^{-1}X = \mathbb{I}_n$. Used for finding solutions, i.e. the weights to solve linear equations. Example: 
$y = Xw$, $y$: target, $X$: feature matrix, $w$: unknown weights,

$X^{-1}y = X^{-1}Xw$

$X^{-1}y = \mathbb{I}_n w$, hence $ w = X^{-1}y$



