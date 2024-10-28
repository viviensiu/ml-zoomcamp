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
