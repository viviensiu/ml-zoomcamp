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

### 5.3 Web Services: Introduction to Flask
* A **web service** is a method used to communicate between electronic devices via the network. 
* There are some methods in web services we can use it to satisfy our problems. Here below we would list some.
    * **GET**: GET is a method used to retrieve files, For example when we are searching for a cat image in google we are actually requesting cat images with GET method.
    * **POST**: POST is the second common method used in web services. For example in a sign up process, when we are submiting our name, username, passwords, etc we are posting our data to a server that is using the web service. (Note that there is no specification where the data goes)
    * **PUT**: PUT is same as POST but we are specifying where the data is going to.
    * **DELETE**: DELETE is a method that is used to request to delete some data from the server.
    * For more information just google the HTTP methods, You'll find useful information about this.
* We can use `Flask` to create a web service (others are also available, e.g. `Django`, `FastAPI`). 
    * First, `pip install flask`.
    * See example [`ping.py`]() for more details on using Flask.
    * In CLI, type `python ping.py` and access the web service using `127.0.0.1:9696/ping`.
* References:
    * [0.0.0.0 v.s. localhost v.s. 127.0.0.1](https://stackoverflow.com/a/20778887/861423)
    * [Top-level script environment](https://docs.python.org/3.9/library/__main__.html)
    * [Flask app.route()](https://flask.palletsprojects.com/en/2.2.x/api/#flask.Flask.route)