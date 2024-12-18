## Emsemble Trees
[Course materials](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/06-trees) and [slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-6-decision-trees-and-ensemble-learning).

### 6.1 Credit Risk Scoring Project
* [Data source](https://github.com/gastonstat/CreditScoring).
* In this session we'll learn about decision trees and ensemble learning algorithms. The questions that we try to address this week are, "What are decision trees? How are they different from ensemble algorithms? How can we implement and fine-tune these models to make binary classification predictions?"
* To be specific, we'll use credit scoring data to build a model that predicts whether a bank should lend loan to a client or not. The bank takes these decisions based on the historical record.
* In the credit scoring classification problem,
    * If the model returns 0, this means, the client is very likely to payback the loan and the bank will approve the loan.
    * If the model returns 1, then the client is considered as a defaulter and the bank may not approval the loan.

### 6.2 Data cleaning and preparation
* Download the data from the given link.
* Reformat categorical columns (`status`, `home`, `marital`, `records`, and `job`) by mapping with appropriate values.
* Replace the maximum value of `income`, `assets`, and `debt` columns with NaNs.
* Replace the NaNs in the dataframe with 0 (will be shown in the next lesson).
* Extract only those rows in the column `status` who are either ok or default as value.
* Split the data in a two-step process which finally leads to the distribution of 60% train, 20% validation, and 20% test sets with random seed to `11`.
* Prepare target variable `status` by converting it from categorical to binary, where `0` represents `ok` and `1` represents `default`.
* Finally delete the target variable from the train/val/test dataframe.

### 6.3 Decision Trees
* Decision Trees are powerful algorithms, capable of fitting complex datasets. The decision trees make predictions based on a bunch of conditions in if/else statements  by splitting a node into two or more sub-nodes.
* While versatile, the decision tree is also prone to overfitting. One of the reason why this algorithm often **overfits** because of its **depth**. It tends to memorize all the patterns in the train data but struggle to performs well on the unseen data (validation or test set).
* **NOTE**: **A large depth implies a very specific set of conditions to lead to a result.** This often causes less examples to be able to meet all those conditions hence it doesn't generalise well.
* To overcome the overfitting problem, we can reduce the complexity of the algorithm by reducing the depth size.
* The decision tree with only a single depth is called decision stump and it only has one split from the root.
* Classes, functions, and methods:
    * `DecisionTreeClassifier`: classification model from `sklearn.tree` class.
    * `max_depth`: hyperparameter to control the depth of decision tree algorithm.
    * `export_text`: method from `sklearn.tree` class to display the text report showing the rules of a decision tree.

### 6.4 Decision Tree Learning Algorithm
* **Terminologies**:
    * `Nodes`: condition nodes that will split into 2 `leaves`.
    * `Leaves`: decision nodes. For a split condition: feature > T (threshold), it splits into true and false leaves.
* **Misclassification rate**: Used for checking how good our predictions are for a particular node. Say if we predict all records with assets <= 4000 would `default` and vice-versa `ok`, we look at how many actual values are same as the predictions in each class `default` and `ok`. If there are 4 predictions that fall under `default` class and it turns out that one actual value did not match this prediction, the misclassification rate would be $\frac{1}{4}$, i.e. the fraction of errors.
* Hence misclassification rate is useful to evaluate the quality of the split at each node.
* **NOTE**: misclassification rate is also called `impurity`.
* To find the best split condition, we set a range of T values and compute the average impurity for each T. The T with the lowest average impurity is the optimal T for best split condition.
* **Finding the best split algorithm**
    * For F in features:
        * Find all thresholds for F.
        * For T in thresholds:
            * Split dataset using "F > T" condition.
            * Compute the impurity of this split.
    * Select the condition with the lowest impurity.
* **Stopping criteria**:
    * Group already pure.
    * Tree reached depth limit.
    * Stop splitting when remaining group too small to split.
* **Decision Tree Algorithm**:
    * Find the best split.
    * Check if max depth is reached.
    * If left is sufficiently large and not pure yet:
        * Repeat for left.
    * If right is sufficiently large and not pure yet:
        * Repeat for right.

### 6.5 Decision Trees Parameter Tuning
* Two hyperparameters of `DecisionTreeClassifier` classifier are important for hyperparameter tuning: `max_depth` and `min_samples_leaf`.
* `min_samples_leaf`: indicates the minimum number of samples required to split an internal node. By setting a minimum number of samples required at a leaf node, we can prevent the tree from creating overfitting by creating branches with very few samples.
* The different combinations of hyperparameters values are fitted into the model and evaluated on the validation set using AUC scores.
* A heatmap could also be used to visualize quickly which AUC score is the best across the hyperparameters' values.

### 6.6 Ensemble Learning and Random Forest
* Emsemble learning: Having an emsemble of trained (differently) models to make a collective decision instead of letting one trained model make a decision.
* Random Forest is an example of ensemble learning where there are a `n_estimators` of decision tree model being used. These decision tree models are trained independently using different subset of features, and their predictions are aggregated to identify the most popular result. 
* Since Random Forest only select a **random subset of features** from the original data to make predictions, hence retraining the same model would result in different predictions and we need to set the `random_state` to keep the predictions constant.
* Important hyperparameters:
    * `n_estimators`: number of decision tree models used.
    * `max_depth`: depth of the trees
    * `min_sample_leaf` : min. number of samples required to split a node.
    * `criterion`: evaluation criteria to evaluation the quality of the split of each node, hence affects the split conditions at each node.
* Classes, functions, and methods:
    * `from sklearn.ensemble import RandomForestClassifier`: random forest classifier from `sklearn` ensemble class.

### 6.7 Gradient Boosting and XGBoost
* Unlike Random Forest where each decision tree trains independently, in the Gradient Boosting Trees, the models are combined **sequentially**, where each model takes the prediction errors made by the previous model and then tries to improve the prediction. This process continues to n number of iterations, and in the end, all the predictions get combined to make the final prediction.
* XGBoost is one of the libraries which implements the gradient boosting technique. To make use of the library, we need to install with `pip install xgboost`. 
* To train and evaluate the xgboost model, we need to wrap our train and validation data into a special data structure from XGBoost which is called `DMatrix`. This data structure is optimized to train XGBoost models faster.
* XGBoost Training Parameters
    * `eta`: learning rate, which indicates how fast the model learns.
    * `max_depth`: to control the size of the trees.
    * `min_child_weight`: to control the minimum size of a child node.
    * `objective`: To specify which problem we are trying to solve, either regression, or classification (binary: `binary:logistic`, or other).
    * `nthread`: used for parallelized training. Usually you specify values equal to the num. of cores your workstation has/able to provision.
    * `seed`: 1, for reproducibility like random states.
    * `verbosity`: `0` for showing errors only. `1`  to show warnings and above, `2` to include info display, if any, during model training.
* Classes, functions, and methods:
    * `xgb.train()`: method to train xgboost model.
    * `xgb_params`: key-value pairs of hyperparameters to train xgboost model.
    * `eval_metric`: evaluation metrics for validation data.
    * `num_boost_round`: equivalent to `n_estimators` in Random Forest, i.e. the num. of tree models used.
    * `verbose_eval`: frequency of printing out the evaluation results during training and validation. E.g. if `verbose_eval = 5` then the results are printed on every 5 rounds. Defaults to printing every round.
    * `%%capture output`: IPython magic command which captures the standard output and standard error of a cell.
* Extracting results from `xgb.train(..)`:
    * In the video we use jupyter magic command `%%capture output` to extract the output of xgb.train(..) method.
    * Alternatively you can use the evals_result parameter of the xgb.train(..). You can pass an empty dictionary in for this parameter and the `train()` method will populate it with the results. The result will be of type `OrderedDict` so we have to transform it to a dataframe. For this, `zip()` can help. 
* References:
    * [Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/release_0.80/tutorials/model.html)
    * [DMatrix v.s. Pandas](https://stackoverflow.com/questions/70127049/what-is-the-use-of-dmatrix)
    * [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)

### 6.8 XGBoost Parameter Tuning
* XGBoost has various tuneable parameters but the three most important ones are:
    * `eta (default=0.3)`: It is also called learning_rate and is used to prevent overfitting by regularizing the weights of new features in each boosting step. range: [0, 1].
    * `max_depth (default=6)`: Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. range: [0, inf].
    * `min_child_weight (default=1)`: Minimum number of samples in leaf node. range: [0, inf].
* Other useful parameter are:
    * `subsample (default=1)`: Subsample ratio of the training instances. Setting it to 0.5 means that model would randomly sample half of the training data prior to growing trees. range: (0, 1]
    * `colsample_bytree (default=1)`: This is similar to random forest, where each tree is made with the subset of randomly choosen features.
    * `lambda (default=1)`: Also called `reg_lambda`. L2 regularization term on weights. Increasing this value will make model more conservative.
    * `alpha (default=0)`: Also called `reg_alpha`. L1 regularization term on weights. Increasing this value will make model more conservative.

### 6.9 Selecting the best model
* We select the final model from decision tree, random forest, or xgboost based on the best auc scores. After that we prepare the df_full_train and df_test to train and evaluate the final model. If there is not much difference between model auc scores on the train as well as test data then the model has generalized the patterns well enough.
* Generally, XGBoost models perform better on tabular data than other machine learning models but the downside is that these model are easy to overfit cause of the high number of hyperparameter. Therefore, XGBoost models require a lot more attention for parameters tuning to optimize them.

### 6.10 Summary
* Decision trees learn if-then-else rules from data. Can overfit easily.
* Finding the best split: select the least impure split. This algorithm can overfit, that's why we control it by limiting the `max depth` and the size of the group (`min_sample_leaf`).
* Random forest is a way of combining multiple decision trees. It should have a diverse set of models to make good predictions. Uses **parallel training**.
* Gradient boosting trains model **sequentially**: each model tries to fix errors of the previous model. XGBoost is an implementation of gradient boosting.

### Homework
* Module 6 [Homework questions](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/cohorts/2024/06-trees/homework.md)
