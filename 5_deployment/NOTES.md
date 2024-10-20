## Deployment
[Course materials](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/05-deployment) and [link for slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-5-model-deployment).

### 5.1 Overview
* Goal: Allow trained model to be deployed into a production environment so that it could predict churn on the fly via web services called by others.
* Steps:
    * Save model into a pickle file.
    * Create a web service using Flask that loads the model file and exposes API endpoints so that the model could be accessed from other services.
    * Create an isolated virtual env using `pipenv` to wrap this web service so it runs independently of other environments.
    * Containerize this virtual env and the web service using `Docker`.
    * Deploy into an AWS Elastic Beanstalk (Optional).

### 5.2 Saving and Loading the Model
* Both trained model and `DictVectorizer` are saved together in a binary file using `pickle`.
* After saving, load the saved model and `DictVectorizer` to test if they work using a random customer data.
* To convert the Jupyter Notebook into a script which could retrain model, we could save the notebook as a Python script `(.py)` under `File` > `Save and Export Notebook as` > `Python` or `Executable Script`. For this module, it is saved as `train.py`.
* We could then refactor the saved script, and separate the load + predict model as `predict.py`.