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
* Upon running these two containers and test for prediction, we should expect connection error. This is because the `gateway` service is unable to communicate with `tf-serving` inside `clothing-model`. In order to connect the two containers and work simultaneously we need docker compose. 
* Docker compose require YAML file which will be executed when run command `docker compose`, usually the file is named as `docker-compose.yaml`:
    ```bash
    version: "3.9"
    services:
        clothing-model: # tf-serving model
            image: zoomcamp-10-model:xception-v4-001
        gateway: # flask gateway service
            image: zoomcamp-10-gateway:002 # new version
            environment:
                - TF_SERVING_HOST=clothing-model:8500 # look for clothing model and port 8500
            ports: # map host machine with gateway
            - "9696:9696"
    ```
* Now we also need to make slight changes in the `gateway.py` to make the environment variable configurable and assign it to the host. This can be done using: 
    ```python
    # the env variable is passed in from the docker-compose.yaml
    # if env variable 'TF_SERVING_HOST' exists, uses its values
    # else use 'localhost:8500'
    host = os.getenv('TF_SERVING_HOST', 'localhost:8500')
    ```
* Running the command `docker-compose up` or detached mode `docker-compose up -d` will establish this connection between both images and everything is configured properly we should have the request predictions.
* We could test if our setup is successful using `python test.py`.
* Useful commands:
    * `docker-compose up`: run docker compose.
    * `docker-compose up -d`: run docker compose in detached mode.
    * `docker ps`: to see the running containers.
    * `docker-compose down`: stop the docker compose.

### Introduction to Kubernetes
* Kubernetes is an open source system for automating deployment scaling and management of containerized applications.
* [Kubernetes Architecture](https://kubernetes.io/docs/concepts/architecture/):
    * `Cluster`: Consists of a control plane plus a set of worker machines, called `nodes`, that run containerized applications. Every cluster needs at least one worker node in order to run Pods.
    * `Nodes`: Think of it as a VM, with one or more pods inside.
    * `Pods`: Your Docker Containers. One pod = one container.
    * `Deployment`: Group of pods with the same set of image and configurations.
    * `Service`: The entrypoint of an application, routes requests to pods depending on current load. For this module, we have 2 services: `gateway service` to handle incoming requests and outgoing response, `model service` to receive gateway requests and respond with predictions back to gateway service. Note that `gateway service` is an external service, while `model service` is internal service.
    * `external service`: load balancer.
    * `internal service`: cluster IP.
    * `ingress`: entrypoint to the cluster.
    * Depending on traffic, kubernetes could scale up by increasing number of pods. This depends on the kubernetes configurations. This is managed by `Horizontal Pod Autoscaler (HPA)`.

### 10.6 ML Zoomcamp 10.6 - Deploying a Simple Web Service to Kubernetes
* **Step 1: Create a simple ping application in Flask**
    * Create a directory `ping`. Create a separate pipenv enironment to avoid conflict by executing `pipenv install flask gunicorn`.
    * From module 5, copy `ping.py` and `Dockerfile` into `ping` directory, modify and then build the image from inside the `ping` directory. The modified `ping.py` is as follow:
    ```python
      # ping.py
    from flask import Flask

    app = Flask('ping-app')

    @app.route('/ping', methods=['GET'])
    def ping():
        return 'PONG'

    if __name__=="__main__":
        app.run(debug=True, host='0.0.0.0', port=9696)
    ```
    * Modified `Dockerfile`:
    ```bash
    # Dockerfile
    FROM python:3.9-slim

    RUN pip install pipenv

    WORKDIR /app

    COPY ["Pipfile", "Pipfile.lock", "./"]

    RUN pipenv install --system --deploy

    COPY "ping.py" .

    EXPOSE 9696

    ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "ping:app"]
    ```
* **Step 2: Install kubectl and kind to build and test cluster locally**
    * **NOTE**: I skipped this step since I already have `kubectl` installed from a previous Udemy course. Also skipped `kind` due to issues reported by other students.
    * We'll install `kubectl` from AWS because later we deploy our application on AWS: 
    ```bash
    curl -o kubectl https://s3.us-west-2.amazonaws.com/amazon-eks/1.24.7/2022-10-31/bin/linux/amd64/kubectl
    ```
    * To install `kind` to setup local kubernetes setup (executable binaries): 
    `wget https://kind.sigs.k8s.io/dl/v0.17.0/kind-linux-amd64 -O kind` > `chmod +x ./kind`. Once the utility is installed we need to place this into our `$PATH` at our preferred binary installation directory.
* **Step 3: Setup kubernates cluster and test it**
    * **Note** Skipped this step.
    * First thing we need to do is to create a cluster: `kind create cluster` (default cluster name is kind).
    * Configure kubectl to interact with kind: `kubectl cluster-info --context kind-kind`.
    * Check the running services to make sure it works: `kubectl get service`.
* **Step 4: Create a deployment**
    * Kubernates requires a lot of configuration and for that VS Code has a [handy extension](https://code.visualstudio.com/docs/azure/kubernetes) that can take a lot of hussle away.
    * Create `deployment.yaml`:
    ```bash
    apiVersion: apps/v1
    kind: Deployment
    metadata: # name of the deployment
        name: ping-deployment
    spec:
        replicas: 1 # number of pods to create
        selector:
            matchLabels: # all pods that have the label app name 'ping' are belonged to 'ping-deployment'
            app: ping
        template: # template of pods (all pods have same configuration)
            metadata:
                labels: # each app gets the same label (i.e., ping in our case)
                    app: ping
            spec: # specs for each pod
                containers: # name of the container
                - name: ping-pod
                  image: ping:v001 # docker image with tag
                  resources:
                    limits:
                      memory: "128Mi"
                      cpu: "500m"
                  ports:
                  - containerPort: 9696 # port to expose
    ```
    * We can now apply the `deployment.yaml` to our kubernetes cluster: `kubectl apply -f deployment.yaml`
    * Next we need to load the docker image into our cluster: `kind load docker-image ping:v001`
    * Executing the command `kubectl get pod` should give the pod status running.
    * To test the pod by specifying the ports: `kubectl port-forward pod-name 9696:9696` and execute `curl localhost:9696/ping` to get the response.
* **Step 5: Create service for deployment**:
    * Create `service.yaml`:
    ```bash
    apiVersion: v1
    kind: Service
    metadata: # name of the service ('ping')
        name: ping
    spec:
        type: LoadBalancer # type of the service (external in this case)
        selector: # which pods qualify for forwarding requests
            app: ping
        ports:
        - port: 80 # port of the service
          targetPort: 9696 # port of the pod
    ```
    * Apply `service.yaml`: `kubectl apply -f service.yaml`
    * Running `kubectl get service` will give us the list of external and internal services along with their service type and other information.
    * Test the service by port forwarding and specifying the ports: `kubectl port-forward service/ping 8080:80` (using 8080 instead to avoid permission requirement) and `executing curl localhost:8080/ping` should give us the output `PONG`.
* **Step 6: Setup and use `MetalLB` as external load-balancer**
    * **Note**: This part is not in the video lecture.
    * Apply MetalLB manifest
    ```bash
    kubectl apply -f https://raw.githubusercontent.com/metallb/metallb/v0.13.7/config/manifests/metallb-native.yaml
    ```
    * Wait until the MetalLB pods (controller and speakers) are ready
    ```bash
    kubectl wait --namespace metallb-system \
             --for=condition=ready pod \
             --selector=app=metallb \
               --timeout=90s
    ```
    * Setup address pool used by loadbalancers:
        * Get range of IP addresses on docker kind network.
        ```bash
        docker network inspect -f '{{.IPAM.Config}}' kind
        ```
        * Create IP address pool using `metallb-config.yaml`.
        ```bash
        apiVersion: metallb.io/v1beta1
        kind: IPAddressPool
        metadata:
            name: example
            namespace: metallb-system
        spec:
            addresses:
            - 172.20.255.200-172.20.255.250
        ---
        apiVersion: metallb.io/v1beta1
        kind: L2Advertisement
        metadata:
            name: empty
            namespace: metallb-system
        ```
    * Apply deployment and service for updates
    ```bash
    kubectl apply -f deployment.yaml
    kubectl apply -f service.yaml
    ```
    * Get external LB_IP
    ```bash
    kubectl get service
    ```
    * Test using load-balancer ip address
    ```bash
    curl <LB_IP>:80/ping
    ```

### 10.7 Deploying TensorFlow Models to Kubernetes
* In previous lesson, we only deployed a basic `ping.py` service to Kubernetes which returns "PONG" whenever we invoke this service. Now we will deploy the full set of gateway + clothing model services to Kubernetes. This is based on the `docker-compose.yaml` created in lesson 10.4. 
* Create a new subfolder `kube-config`.
* **Step 1: Create deployment for the tf-serving model `model-deployment.yaml`**
    ```bash
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: tf-serving-clothing-model
    spec:
        replicas: 1 # num of pods
        selector:
            matchLabels:
              app: tf-serving-clothing-model
        template:
            metadata:
              labels:
                app: tf-serving-clothing-model
            spec:
                containers:
                - name: tf-serving-clothing-model
                image: zoomcamp-10-model:xception-v4-001 # if use kind, make this img available in kind
                resources:
                    limits:
                        memory: "512Mi"
                        cpu: "0.5"
                ports:
                - containerPort: 8500
    ```
    * Load the model image to kind: `kind load docker-image clothing-model:xception-v4-001`
    * Create model deployment: `kubectl apply -f model-deployment.yaml`
    * Get the running pod id for the model: `kubectl get pod`
    * Test the model deployment using the pod id: `kubectl port-forward tf-serving-clothing-model-85cd6dsb6-rfvg410m 8500:8500` and run `gateway.py` script to get the predictions.
* **Step 2: Create service of tf-serving model model-service.yaml**
    ```bash
    apiVersion: v1
    kind: Service
    metadata:
        name: tf-serving-clothing-model
    spec:
        type: ClusterIP # default service type is always ClusterIP (i.e., internal service)
        selector:
            app: tf-serving-clothing-model
        ports:
        - port: 8500
            targetPort: 8500
    ```
    * Create model service: `kubectl apply -f mdoel-service.yaml`
    * Check the model service: `kubectl get service`.
    * Test the model service: `kubectl port-forward service/tf-serving-clothing-model 8500:8500` and run `gateway.py` for predictions.
* **Step 3: Create deployment for the gateway gateway-deployment.yaml**
    ```bash
    apiVersion: apps/v1
    kind: Deployment
    metadata:
    name: gateway
    spec:
    selector:
        matchLabels:
        app: gateway
    template:
        metadata:
        labels:
            app: gateway
        spec:
        containers:
        - name: gateway
            image: zoomcamp-10-gateway:002
            resources:
            limits:
                memory: "128Mi"
                cpu: "100m"
            ports:
            - containerPort: 9696
            env: # set the environment variable for model
            - name: TF_SERVING_HOST
                value: tf-serving-clothing-model.default.svc.cluster.local:8500 # kubernetes naming convention: <NAME>.default.svc.cluster.local:<PORT>
    ```
    * Load the gateway image to kind: `kind load docker-image clothing-model-gateway:002`
    * Create gateway deployment `kubectl apply -f gateway-deployment.yaml` and get the running pod id `kubectl get pod`
    * Test the gateway pod: `kubectl port-forward gateway-6b945f541-9gptfd 9696:9696` and execute `test.py` for get predictions.
* Create service of tf-serving model `gateway-service.yaml`:
    ```bash
    apiVersion: v1
    kind: Service
    metadata:
    name: gateway
    spec:
        type: LoadBalancer # External service to communicate with client (i.e., LoadBalancer)
        selector:
            app: gateway
        ports:
        - port: 80 # port of the service
            targetPort: 9696 # port of load balancer
    ```
    * Create gateway service: `kubectl apply -f gateway-service.yaml`
    * Get service id: `kubectl get service`
    * Test the gateway service: `kubectl port-forward service/gateway 8080:80` and replace the url on `test.py` to 8080 to get predictions.











