## Evaluation
Course materials [here](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/04-evaluation). [Link for slides](https://www.slideshare.net/slideshow/ml-zoomcamp-4-evaluation-metrics-for-classification/250301340).

### 4.1 Evaluation: Session overview
The fourth week of Machine Learning Zoomcamp is about different metrics to evaluate a binary classifier. These measures include accuracy, confusion table, precision, recall, ROC curves(TPR, FPR, random model, and ideal model), AUC-ROC, and cross-validation.

### 4.2 Accuracy and Dummy model
* Accuracy: fraction of correct prediction
* To see if we have the optimal accuracy at 0.5 threshold, we evaluate different accuracies on churn thresholds at 0, 0.05, 0.10 .... 0.95, 1. 
* **Dummy model**: model that gives a constant prediction, such as all true or all false values.
* Notice that when we make a dummy prediction that all records will not churn, i.e. churn = 0, the accuracy equals 72.6%, which is not too different from 80.2% (where the threshold is set at 0.5). In other words, there is not much of an improvement from making a dummy prediction compared to using a threshold.
* Classes and methods:
    * `np.linspace(x,y,z)` - returns a numpy array starting at `x` until `y` with `z` evenly spaced samples
    * `Counter(x)` - collection class that counts the number of instances that satisfy the x condition
    * `accuracy_score(x, y)` - sklearn.metrics class for calculating the accuracy of a model, given a predicted `x` dataset and a target `y` dataset.

### Confusion Matrix
* True positive (TP): both predictions and actual value are positive.
* True negative (TN): both predictions and actual value are negative.
* False positive (FP): predicted positive but actual value is negative.
* False negative (FN): predicted negative but actual value is positive.
* The accuracy corresponds to the sum of TN and TP divided by the total of observations. But accuracy doesn't tell us much when there's **class imbalance**, as seen before when we try to predict using dummy model.
* Structure of confusion matrix:

|  predict=negative | predict=positive | |
|----------|----------|-|
| TN | FP | actual=negative |
| FN | TP | actual=positive |

### Precision & Recall
* Precision: fraction of positive examples that are correctly identified where $\text{ precision} = \displaystyle\frac{TP}{\text{predicted positives}} = \frac{TP}{TP+FP}$
* Recall: fraction of true positives that are correctly identified where $\text{ recall} = \displaystyle\frac{TP}{\text{actual positives}} = \frac{TP}{TP+FN}$
* Accuracy: $\text{ accuracy} = \displaystyle\frac{TP+TN}{TP+TN+FP+FN}$
* It's important to figure out the precision and recall because for our use case, although the accuracy is 80%, the precision is only 67% and recall is 54%.
* This implies that for precision = 67%, 33% customers that would not churn are getting promotional emails.
* For recall = 54%, it means 46% of customers that would churn are not getting promotional emails, which doesn't help since the promotional campaign is missing 46% of its target customers! 
* So, these measures reflect some errors of our model that accuracy did not notice due to the class imbalance. 
* **MNEMONICS**:
    * Precision : From the `pre`dicted positives, how many we predicted right. See how the word `pre`cision is similar to the word `pre`diction?
    * Recall : From the `real` positives, how many we predicted right. See how the word `re`call is similar to the word `real`?

