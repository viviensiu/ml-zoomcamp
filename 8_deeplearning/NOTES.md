## Deep Learning
[Course materials](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/08-deep-learning) and [slides](https://www.slideshare.net/slideshow/ml-zoomcamp-8-neural-networks-and-deep-learning-250592316/250592316).

### Setup
* `pipenv install tensorflow`.

### 8.1 Fashion classification
* This module would focus on using deep learning to perform multiclass image classification. The deep learning frameworks like TensorFlow and Keras will be implemented on clothing dataset to classify images of t-shirts.
* Dataset used: [clothing dataset (small)](https://github.com/alexeygrigorev/clothing-dataset-small) which contains 10 of the fashion classification classes. Use `git clone https://github.com/alexeygrigorev/clothing-dataset-small.git` to clone this dataset.

### 8.2 Tensorflow and Keras
* Tensorflow: Deep learning framework.
* Keras: high-level abstraction that sits on top on Tensorflow.
* **Note**: Instructions to setup GPU is not part of this module.
* To train/use an image recognition model using Keras requires images to be loaded in predefined dimensions. This is specified in `target_size` parameter of the `load_img` function.
* Each image consists of pixels, each pixel is made of 3 color channels: Red, Green, Blue. Each color channel has values ranged from 0 to 255, 0 being black color.
* Hence an image dimension is of (height, width, channels) e.g. $(150 \times 150 \times 3)$.
* The image is represented as a PIL object and also can be loaded in numpy array, its values are of uint8 (unsigned integer 8 bit)
* Classes, functions, and methods:
    * `import tensorflow as tf`: to import tensorflow library
    * `from tensorflow import keras`: to import keras
    * `from tensorflow.keras.preprocessing.image import load_img`: to import load_img function
    * `load_img('path/to/image', targe_size=(150,150))`: to load the image of 150 x 150 size in PIL format
    * `np.array(img)`: convert image into a numpy array of 3D shape, where each row of the array represents the value of red, green, and blue color channels of one pixel in the image.

### 8.3 Pretrained Convolutional Neural Network
* Keras provides a number of pretrained models which are mainly trained on [ImageNet](https://www.image-net.org/). [See here for all the Keras pretrained models](https://keras.io/api/applications/)
* A pretrained model `Xception` will be used here. To do so,
    * `from tensorflow.keras.applications.xception import Xception`: import the model from keras applications.
    * `from tensorflow.keras.application.xception import preprocess_input`: function to perform preprocessing on images.
    * `from tensorflow.keras.applications.xception import decode_predictions`: extract the predictions class name in the form of tuple of list.
*  `preprocess_input` accepts a batch of images, hence the batch needs to be defined as [num. of images in batch, **dimensions of each image], for example if you have 5 images of (299, 299, 3), the batch of images should be `X = numpy.array([5, 299, 299, 3])`.
* Steps:
    * Initialize `Xception` model.
    * Pass the set of unprocessed images to `preprocess_input` to get the processed images `X`.
    * Predict using `model.predict(X)` to get an array of predictions: Each row is a prediction for an image, each column represents the prediction classes. E.g. a prediction of shape (1,1000) means a single image prediction of 1000 possible image classes.

### 8.4 Convolutional Neural Network
* What is Convolutional Neural Network?: 
> A convolutional neural network, also know as CNN or ConvNet, is a feed-forward neural network that is generally used to analyze viusal images by processing data with grid-like topology. A CNN is used to detect and classify objects in an image. In CNN, every image is represented in the form of an array of pixel values.
The convolution operation forms the basis of any CNN. In convolution operation, the arrays are multiplied element-wise, and the dot product is summed to create a new array, which represents `Wx`.
* Layers in a Convolutional Neural Network: A Convolution neural network has multiple hidden layers that help in extracting information from an image. The four important layers in CNN are:
    * Convolution layer
    * Activation function layer
    * Pooling layer
    * Fully connected layer (also called Dense layer)
* **Convolution layer**: 
    * Extracts valuable features from an image. A convolution layer has several filters that perform the convolution operation. Every image is considered as a matrix of pixel values.
    * Consider a black and white image of 5x5 size whose pixel values are either 0 or 1 and also a filter matrix with a dimension of 3x3. Next, slide the filter matrix over the image and compute the dot product to get the convolved feature matrix.
* **Activation function layer**:
    * Once the feature maps are extracted, the next step is to move them to a activation function layer. There are different activation functions such as ReLU, Sigmoid, Softmax etc.
    * **ReLU (Rectified Linear Unit)** is an activation function which performs an element-wise operation and sets all the negative pixels to 0. It introduces non-linearity to the network, and the generated output is a rectified feature map. The relu function is: $f(x) = \max(0,x)$.
    * **Sigmoid**: for binary classification. Values for each neuron in this layer sums to 1.
    * **Softmax**: for multi-class classification. Values for each neuron in this layer sums to 1.
* **Pooling layer**:
    * A down-sampling operation that reduces the dimensionality of the feature map. The rectified feature map goes through a pooling layer to generate a pooled feature map.
    * Imagine a rectified feature map of size 4x4 goes through a max pooling filter of 2x2 size with stride of 2. In this case, the resultant pooled feature map will have a pooled feature map of 2x2 size where each value will represent the maximum value of each stride.
    * The pooling layer uses various filters to identify different parts of the image like edges, shapes etc.
* **Fully Connected layer (Dense layer)**:
    * The next step in the process is called flattening. Flattening is used to convert all the resultant 2D arrays from pooled feature maps into a single linear vector. This flattened vector is then fed as input to the fully connected layer to classify the image.
    * `fully-connected` due to every neuron in the flattened layer is connected to every neuron in the next layer.
* **CNN in a nutshell**:
    * The pixels from the image are fed to the convolutional layer that performs the convolution operation.
    * It results in a convolved map.
    * The convolved map is applied to a ReLU function to generate a rectified feature map.
    * The image is processed with multiple convolutions and ReLU layers for locating the features.
    * Different pooling layers with various filters are used to identify specific parts of the image.
    * The pooled feature map is flattened and fed to a fully connected layer to get the final output.
* References: [Learn CNN in the browser](https://poloclub.github.io/cnn-explainer/).

### 8.5 Transfer Learning
* A pretrained model is usually trained to predict a certain number of outputs.
* For Keras Pretrained models, the CNN layers up till the vector representation are trained on general image classification filters and such. However in the Dense layers, these models are trained to predict the 1000 image classes as made available in ImageNet dataset.
* For use cases which we want a model to make different predictions than the 1000 classes above but we want to keep the generic CNN layers that are quite useful in the model, we could use a method called **Transfer Learning**. 
* What **Transfer Learning** does is it keeps the CNN layers together with the filters up till vector representation layer, but discard the Dense layers so we could customise the model to perform predictions that apply to our own use cases.
* Note that the **Transfer Learning** model needs to be retrained with our own specific dataset so it could make predictions pertaining to our use cases.
```python
# Build image generator for training (takes preprocessing input function)
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load in train dataset into train generator
train_ds = train_gen.flow_from_directory(directory=path/to/train_imgs_dir, # Train images directory
                                         target_size=(150,150), # resize images to train faster
                                         batch_size=32) # 32 images per batch

# Create image generator for validation
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load in image for validation
val_ds = val_gen.flow_from_directory(directory=path/to/val_imgs_dir, # Validation image directory
                                     target_size=(150,150),
                                     batch_size=32,
                                     shuffle=False) # False for validation
```
