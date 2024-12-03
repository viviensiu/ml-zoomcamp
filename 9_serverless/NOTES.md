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