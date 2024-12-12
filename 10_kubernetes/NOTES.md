## Kubernetes
[Course materials](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/10-kubernetess) and [slides](https://www.slideshare.net/slideshow/ml-zoomcamp-10-kubernetes/250763271).

### 10.1 Overview
* **Goal**: Separate the functionality for the clothing prediction application into 2 containers and deploy them using Kubernetes.
* In module 9, we were using `tflite` for serving, `tflite` is written in C++, with focus on inference.
* Tensorflow serving uses gRPC binary protocol due to its efficiency in data exchange.
* To deploy to Kubernetes, we separate the current workflow into 2 components:
    * 1st component: Gateway (download image, resize, turn into numpy array - computationally not expensive - can be done with CPU).
    * 2nd component: Model (matrix multiplications - computationally expensive - thus use GPU).
* This allows scaling the two components independently: i.e. 5 gateways handing images to 1 model.

### 10.2 Tensorflow Serving
**Step 1: Convert model to SavedModel**

* To build the app we need to convert the keras model `HDF5` into special format called tensorflow `SavedModel`. To do that, we download a prebuilt model and save it in the working directory:
    ```bash
    wget https://github.com/DataTalksClub/machine-learning-zoomcamp/releases/download/chapter7-model/xception_v4_large_08_0.894.h5 -O clothing-model.h5
    ```
* Then convert the model to `SavedModel` format:
    ```python
    import tensorflow as tf
    from tensorflow import keras

    model = keras.models.load_model('./clothing-model.h5')

    tf.saved_model.save(model, 'clothing-model')
    ```
* We can inspect what's inside the saved model using the utility (saved_model_cli) from TensorFlow and the following command:
    ```bash
    saved_model_cli show --dir clothing-model --all
    ```
* Running the command outputs a few things but we are interested in the signature, specifically the following one. For instance:
    ```bash
    signature_def['serving_default']:
    The given SavedModel SignatureDef contains the following input(s):
        inputs['input_8'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1, 299, 299, 3)
            name: serving_default_input_8:0
    The given SavedModel SignatureDef contains the following output(s):
        outputs['dense_7'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1, 10)
            name: StatefulPartitionedCall:0
    Method name is: tensorflow/serving/predict
    ```
* Alternatively one can also use the following command to output just the desired part:
    ```bash
    saved_model_cli show --dir clothing-model --tag_set serve --signature_def serving_default
    ```
* Make a note for `serving_default` as we need these info later on for serving:
    ```
    serving_default
    input_8 - input
    dense_7 - output
    ```

**Step 2: Serve the model locally from Docker Container**
* We can run the model (clothing-model) with the prebuilt docker image tensorflow/serving:2.7.0:
    ```bash
    docker run -it --rm \
    -p 8500:8500 \
    -v $(pwd)/clothing-model:/models/clothing-model/1 \
    -e MODEL_NAME="clothing-model" \
    tensorflow/serving:2.7.0
    ```
    where:
    * `docker run -it --rm`: to run the docker container in interactive mode and removes it once stopped.
    * `-p 8500:8500`: port mapping, must match the ports defined in app.
    * `-v $(pwd)/clothing-model:/models/clothing-model/1`: volume mapping of absolute model directory to model directory inside the docker image.
    * `-e MODEL_NAME="clothing-model"`: set environment variable for docker image.
    * `tensorflow/serving:2.7.0`: name of the image to run.

**Step 3: Invoke the served model using Jupyter Notebook**
* Tensorflow uses specical serving called `gRPC` protocol which is optimized to use binary data format. We need to convert our prediction into `protobuf`.
* See `tf-serving-connect.ipynb` for subsequent setups.


### 10.3 Creating a pre-processing service
* In the previous section we created jupyter notebook to communicates with the model deployed with tensorflow. This notebook fetches an image, pre-process it, turns it into protobuf, sends it to tensorflow-serving, does post-processing, and finally gives a human-readable answer.
* In this section we convert the notebook `tf-serving-connect.ipynb` into python script `gateway.py` to build flask application. To convert the notebook into script we can run the command `jupyter nbconvert --to script notebook.ipynb` and we rename the script to `gateway.py`.
* If you do not have Flask installed, do `pipenv install flask`.
* Then we create functions to prepare request, send request, and prepare response. For flask app we can reuse the code from session 5:
    ```python
    # Create flask app
    app = Flask('gateway')

    @app.route('/predict', methods=['POST'])
    def predict_endpoint():
        data = request.get_json()
        url = data['url']
        result = predict(url)
        return jsonify(result)
    ```
* Our application has two components: docker container with tensorflow serving and flask application with the gateway.
* We also want to put everything in the pipenv for deployment. For that we need to install few libraries with pipenv: `pipenv install grpcio==1.42.0 gunicorn keras-image-helper`.
* As we discussed, tensorflow is a large library and we don't want to use it in our application. Instead we can use the following script to convert numpy array into protobuf format and import the `np_to_protobuf` function into our `gateway.py` script. In order the make the script work we need to install the following libraries as well `pipenv install tensorflow-protobuf==2.7.0 protobuf==3.19`:
    ```python
    from tensorflow.core.framework import tensor_pb2, tensor_shape_pb2, types_pb2

    def dtypes_as_dtype(dtype):
        if dtype == "float32":
            return types_pb2.DT_FLOAT
        raise Exception("dtype %s is not supported" % dtype)

    def make_tensor_proto(data):
        shape = data.shape
        dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=i) for i in shape]
        proto_shape = tensor_shape_pb2.TensorShapeProto(dim=dims)

        proto_dtype = dtypes_as_dtype(data.dtype)

        tensor_proto = tensor_pb2.TensorProto(dtype=proto_dtype, tensor_shape=proto_shape)
        tensor_proto.tensor_content = data.tostring()

        return tensor_proto

    def np_to_protobuf(data):
        if data.dtype != "float32":
            data = data.astype("float32")
        return make_tensor_proto(data)
    ```
* Bash script to create custom tf-serving-protobuf and compile: [https://github.com/alexeygrigorev/tensorflow-protobuf/blob/main/tf-serving-proto.sh](https://github.com/alexeygrigorev/tensorflow-protobuf/blob/main/tf-serving-proto.sh).

### 10.4 Docker Compose
* Docker Compose is a tool that help us to define and share multi-container applications. With Compose, we can create a YAML file to define the services (in our case it is `gateway` service and `clothing-model` model) and with a single command, we can spin everything up inside the YAML file or tear it all down with just one command. Docker compose is very useful the test the applications locally.
* Instead of mapping the volume, port, and run the docker container in the terminal for our tf-serving model (`clothing-model`), we want to create docker image and put everything in there. For this we want to create docker image by the name `image-model.dockerfile`:
    ```bash
    FROM tensorflow/serving:2.7.0

    # Copy model in the image
    COPY clothing-model /models/clothing-model/1
    # Specify environmental variable
    ENV MODEL_NAME="clothing-model"
    ```
* Next we build its image. We need to specify the dockerfile name along with the tag since it's not named `Dockerfile`: 
    ```bash
    docker build -t clothing-model:xception-v4-001 -f image-model.dockerfile .
    ```
* Next we can run this built image with:
    ```bash
    docker run -it --rm -p 8500:8500 clothing-model:xception-v4-001
    ```
* Similarly we can do the same thing for our `gateway` service. The file name is `image-gateway.dockerfile`:
    ```bash
    FROM python:3.8.12-slim

    RUN pip install pipenv

    # Create working directory in docker image
    WORKDIR /app

    # Copy Pipfile and Pipfile.lock files in working dir
    COPY ["Pipfile", "Pipfile.lock", "./"]

    # Install required packages using pipenv
    RUN pipenv install --system --deploy

    # Copy gateway and protobuf scripts in the working dir
    COPY ["gateway.py", "protobuf.py", "./"]

    EXPOSE 9696

    ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "gateway:app"]
    ```
    * Build image: `docker build -t clothing-model-gateway:001 -f image-gateway.dockerfile .` 
    * Run image: `docker run -it --rm -p 9696:9696 clothing-gateway:001`
*





