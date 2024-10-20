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
    * See example [`ping.py`](https://github.com/viviensiu/ml-zoomcamp/blob/main/5_deployment/ping.py) for more details on using Flask.
    * In CLI, type `python ping.py` and access the web service using `127.0.0.1:9696/ping`.
    * Press ctrl+C to quit running the web service.
* References:
    * [0.0.0.0 v.s. localhost v.s. 127.0.0.1](https://stackoverflow.com/a/20778887/861423)
    * [Top-level script environment](https://docs.python.org/3.9/library/__main__.html)
    * [Flask app.route()](https://flask.palletsprojects.com/en/2.2.x/api/#flask.Flask.route)

### 5.4 Serving the Churn Model with Flask
* See [`predict.py`](https://github.com/viviensiu/ml-zoomcamp/blob/main/5_deployment/predict.py) for detailed notes on `request` and `jsonify`.
* Once app is served using `python predict.py`, you could post a customer info in JSON format to make predictions using either another script, a Jupyter notebook or [Postman API](https://www.postman.com/).
* Note that Flask by default is served in development server and you will see a warning `WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.`.
* Hence, one way to serve in production is by using `gunicorn`: `pip install gunicorn`.
* Then, serve using gunicorn by binding the url to the Flask app: `gunicorn --bind 0.0.0.0:9696 predict:app`. You could retest prediction as per point 2 above.
* **NOTE**: `gunicorn — bind` means it is being attached to a URL that the application can be reached. In this case, the many zeroes or 0.0.0.0 means localhost, and the 9696 is the port number specified in `predict.py` > `app.run()`. The application can be reached locally at `localhost:9696`. Lastly, the name of the flask_file followed by `:app` must be specified for Gunicorn to know, what Python files are being run.
* **NOTE**: Stop Flask app before serve the app using `gunicorn`.
* References:
    * [flask.request to process incoming JSON from POST method](https://tedboy.github.io/flask/generated/generated/flask.Request.html)
    * [flask.jsonify to send response in JSON format](https://tedboy.github.io/flask/generated/flask.jsonify.html)
    * [gunicorn: Python Web Server Gateway Interface (WSGI)](https://gunicorn.org/)
    * [How to Run Gunicorn in the Python Microframework Flask](https://medium.com/@andrewdass/how-to-run-gunicorn-in-the-python-microframework-flask-32a41abe2755)