## Serverless
[Course materials](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/09-serverless).

### 9.1 Introduction to Serverless
* This module explores how to deploy a deep learning model to the cloud, specifically using aws lambda and tensorflow lite.

### 9.2 AWS Lambda
* AWS Lambda is pay-per-trigger hence it is cheaper to keep running compared to deploying the model to an EC2 instance.
* Basically you create a Python function using AWS Lambda.
* The `event` in the function are JSON requests to your function, and your function will also need to respond with a JSON response.

### 9.3 Tensorflow Lite (Tflite)
* A smaller version of Tensorflow that helps with deploying to AWS Lambda as:
    * Larger image = pay more for storage.
    * Larger image = longer initialization.
    * Larger image = higher RAM utilization. 
* Tflite only focus on **inferencing, i.e. predicting**. You need to build and train your model with Tensorflow, and only convert the model to a tflite version once you're happy with the model.
* Conversion:
```python
# Converting a tf.Keras model to a TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```
* Load tflite model:
```python
# load tflite model
interpreter = tflite.Interpreter(model_path="model.tflite")
# assign weights to the tflite model
interpreter.allocate_tensors()
# get the index that expects input within interpreter
input_index = interpreter.get_input_details()[0]['index]
# get the index that returns output within interpreter
output_index = interpreter.get_output_details()[0]['index]
# assigns preprocessed image for prediction
interpreter.set_tensor(input_index, X)
# process input
interpreter.invoke()
# gets prediction
preds = interpreter.get_tensor(output_index)
dict(zip(classes, preds[0]))
```
* Classes, functions and methods:
    * `import tensorflow.lite as tflite`:
    * `tensorflow.lite.TFLiteConverter.from_keras_model()`: tflite converter to convert a Keras model
    * `.convert()`: converts to tflite model.
    * `tflite.Interpreter()`: interpreter for tflite models
    * `interpreter.allocate_tensors()`: allocate weight tensors to tflite model
    * `interpreter.get_input_details()`: returns a dictionary of input information. 
    * `interpreter.get_output_details()`: returns a dictionary of output information. 
    * `interpreter.set_tensor()`: provides input to tflite model neuron, used for providing input. 
    * `interpreter.get_tensor()`: provides output from tflite model output neurons, used for retrieving predictions. 
* References:
    * [Notes from Peter Ernicke on "Excluding TensorFlow dependency"](https://knowmledge.com/2023/12/02/ml-zoomcamp-2023-serverless-part-3/)

### 9.4 Preparing the Code for AWS Lambda
* Converts code from Module 9.3 into a Python script that can be called in command line.
* `predict()`: Accepts a url and makes a prediction.
* `lambda_function()`: The function to be triggered by AWS Lambda event, accepts a JSON request which contains a url, and returns a JSON response containing the predictions.

### 9.5 Preparing a Docker Image
**Python 3.12 vs TF Lite 2.17**
* The latest versions of TF Lite don't support Python 3.12 yet.
* As a workaround, we can use the previous version of TF Lite to serve the models created by TensorFlow 2.17. We tested it with TF Lite 2.14 and the deep learning models we use in the course work successfully with this setup.
* Here's how you do it:
    * First, use Python 3.10. It means that you will need to use `public.ecr.aws/lambda/python:3.10` as the base image:
    * ```FROM public.ecr.aws/lambda/python:3.10```.
    * Second, use numpy 1.23.1:
    * ```RUN pip install numpy==1.23.1```
    * When installing tf lite interpreter for AWS lambda, make sure you don't install dependencies with `--no-deps` flag:
    * ```RUN pip install --no-deps https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl```
    * If you don't do it, pip will try to upgdate the version of numpy and your code won't work (as the tflite runtime was compiled with numpy 1, not numpy 2).

**Using pip install for TF-Lite binaries**
* When using pip to install the compiled binary, make sure you use the raw file, not a link to the github page.
* Correct:
```
pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
```
(Note /raw/ in the path)

* Also correct:
```
pip install https://github.com/alexeygrigorev/tflite-aws-lambda/blob/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl?raw=true
```
The wheel file above is for Python 3.10. Check other available compiled TF lite versions here.

* Not correct - won't work:
```
pip install https://github.com/alexeygrigorev/tflite-aws-lambda/blob/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
```
If the file is incorrect, you'll get an error message like that:

zipfile.BadZipFile: File is not a zip file