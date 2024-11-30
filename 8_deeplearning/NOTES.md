## Deep Learning
[Course materials](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/08-deep-learning) and [slides](https://www.slideshare.net/slideshow/ml-zoomcamp-8-neural-networks-and-deep-learning-250592316/250592316).

### Setup
* `pipenv install tensorflow`.

### Saturn Cloud Setup 
* For the purpose of utilising GPU on Saturn Cloud for this module, courtesy of [Saturn Cloud](https://saturncloud.io/).
* Refer [Setup notes](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/08-deep-learning/01b-saturn-cloud.md). 

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
* Steps to create train/validation data for model:
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
* Steps to build model from a pretrained model, this focus on the Dense layers for custom predictions, note that at this step there's no training yet:
```python
# Build base model
base_model = Xception(weights='imagenet',
                      include_top=False, # to create custom dense layer
                      input_shape=(150,150,3))

# Freeze the convolutional base by preventing the weights being updated during training
base_model.trainable = False

# Define expected image shape as input
inputs = keras.Input(shape=(150,150,3))

# Feed inputs to the base model
base = base_model(inputs, training=False) # set False because the model contains BatchNormalization layer

# Convert matrices into vectors using pooling layer
vectors = keras.layers.GlobalAveragePooling2D()(base)

# Create dense layer of 10 classes for predictions
# Note: NO ACTIVATION FUNCTION HERE, as we will include activation during training instead,
# see next step for training
outputs = keras.layers.Dense(10)(vectors)

# Create model for training, takes in inputs and returns outputs as predictions
model = keras.Model(inputs, outputs)
```
* Steps to instantiate optimizer and loss function to train the model:
```python
# Define learning rate
learning_rate = 0.01

# Create optimizer. Used for learning the weights during training
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# Define loss function
loss = keras.losses.CategoricalCrossentropy(from_logits=True) 
# `from_logits=True`: to keep the raw output of dense layer without applying softmax
# In the previous code block, notice that we did not use `activation` in Dense layer, 
# hence the outputs are raw logits instead of probability outputs (e.g. softmax).

# Compile the model
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy']) # evaluation metric accuracy
```
* The model is ready to train once it is defined and compiled:
```python
# Train the model, validate it with validation data, and save the training history
history = model.fit(train_ds, epochs=10, validation_data=val_ds)
```
* Classes, function, and attributes:
    * `from tensorflow.keras.preprocessing.image import ImageDataGenerator`: to read the image data and make it useful for training/validation.
    * `flow_from_directory()`: method to read the images directly from the directory.
    * `next(train_ds)`: to unpack features and target variables.
    * `train_ds.class_indices`: attribute to get classes according to the directory structure.
    * `GlobalAveragePooling2D()`: accepts 4D tensor as input and operates the mean on the height and width dimensionalities for all the channels and returns vector representation of all images.
    * `CategoricalCrossentropy()`: method to produces a one-hot array containing the probable match for each category in multi classification.
    * `epochs`: number of iterations over all of the training data.
    * `history.history`: history attribute is a dictionary recording loss and metrics values (accuracy in our case) for at each epoch.
* References: [Keras Optimizers](https://keras.io/api/optimizers/)

### 8.6 Adjusting the Learning Rate
* It's a tuning parameter in an optimization function that determines the **step size** (how big or small) at each iteration while moving toward a mininum of a loss function.
* We can experiement with different learning rates to find the optimal value where the model has best results. In order to try different learning rates, we should define a function to create a function first, for instance:
```python
# Function to create model
def make_model(learning_rate=0.01):
    base_model = Xception(weights='imagenet',
                          include_top=False,
                          input_shape=(150,150,3))

    base_model.trainable = False
    
    #########################################
    
    inputs = keras.Input(shape=(150,150,3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    outputs = keras.layers.Dense(10)(vectors)
    model = keras.Model(inputs, outputs)
    
    #########################################
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    
    return model
```
* Next, we can loop over on the list of learning rates:
```python
# Dictionary to store history with different learning rates
scores = {}

# List of learning rates
lrs = [0.0001, 0.001, 0.01, 0.1]

for lr in lrs:
    print(lr)
    
    model = make_model(learning_rate=lr)
    history = model.fit(train_ds, epochs=10, validation_data=val_ds)
    scores[lr] = history.history
    
    print()
    print()
```
* Visualizing the training and validation accuracies help us to determine which learning rate value is the best for for the model. One typical way to determine the best value is by looking at the **gap between training and validation accuracy**. The **smaller gap** indicates the optimal value of the learning rate, i.e. less overfitting.

### 8.7 Model Checkpointing
* When training the model, you may notice that the validation accuracies oscillate from one epoch to another.
* Instead of saving models from the whole history, we only want to save the best models so far at that epoch.
* Hence, the solution here is to add checkpoints that would save the current model if it's the <i>best model so far</i>, a.k.a up till the current epoch. 
* In Keras, `ModelCheckpoint` callback is used with training the model to save a model or weights in a checkpoint file at some interval, so the model or weights can be loaded later to continue the training from the state saved or to use for deployment.
* Checkpoint conditions may include reaching the best performance.
* Classes, function, and attributes:
    * `keras.callbacks.ModelCheckpoint`: ModelCheckpoint class from keras callbacks api
    * `filepath`: path to save the model file.
    * `monitor`: the metric name to monitor.
    * `save_best_only`: only save when the model is considered the best according to the metric provided in monitor.
    * `model`: overwrite the save file based on either maximum or the minimum scores according the metric provided in monitor.

### 8.8 Adding More Layers
* It is also possible to add more layers between the `vector representation layer`(`base_model`) and the `output layer` to perform intermediate processing of the vector representation. 
* These layers are the same dense layers as the output, but the difference is that these layers use `relu` activation function for non-linearity.
* Like learning rates, we should also experiment with different values of inner layer sizes:
```python
# Function to define model by adding new dense layer
def make_model(learning_rate=0.01, size_inner=100): # default layer size is 100
    base_model = Xception(weights='imagenet',
                          include_top=False,
                          input_shape=(150,150,3))

    base_model.trainable = False
    
    #########################################
    
    inputs = keras.Input(shape=(150,150,3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    # add `inner` layer with relu activation
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors) # activation function 'relu'
    # change `outputs` layer to accept inner layer as input instead of `vectors` layer
    outputs = keras.layers.Dense(10)(inner)
    model = keras.Model(inputs, outputs)
    
    #########################################
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    
    return model
```
* Next, train the model with different sizes of `inner` layer:
```python
# Experiement different number of inner layer with best learning rate
# Note: We should've added the checkpoint for training but for simplicity we are skipping it
learning_rate = 0.001

scores = {}

# List of inner layer sizes
sizes = [10, 100, 1000]

for size in sizes:
    print(size)
    
    model = make_model(learning_rate=learning_rate, size_inner=size)
    history = model.fit(train_ds, epochs=10, validation_data=val_ds)
    scores[size] = history.history
    
    print()
    print()
```
* **Note**: It may not always be possible that the model improves. Adding more layers mean introducing complexity in the model, which may not be recommended in some cases.
* In the video, adding `inner` layer did not improve the model.

### 8.9 Regularization and Dropout
* Training the same data set over more and more epochs would result in overfitting the data as the model may associate some specific patterns from the dataset with the target and unable to generalise, e.g. t-shirt with round logos.
* To mitigate this, the `dropout` method is used.
* `Dropout` involves randomly removing some neurons during each epoch so the model do not learn any information about these dropped neurons.
* It is akin to "hide" partial information on the dataset so the model do not overfit on the training data and unable to generalise.
* Code example to apply `drop` layer in the neural network architecture, notice that learning rate `0.01` and layer size of `100` is used here:
```python
# Function to define model by adding new dense layer and dropout
def make_model(learning_rate=0.01, size_inner=100, droprate=0.5):
    base_model = Xception(weights='imagenet',
                          include_top=False,
                          input_shape=(150,150,3))

    base_model.trainable = False
    
    #########################################
    
    inputs = keras.Input(shape=(150,150,3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    # droprate: probability to drop neurons
    drop = keras.layers.Dropout(droprate)(inner) # add dropout layer
    outputs = keras.layers.Dense(10)(drop)
    model = keras.Model(inputs, outputs)
    
    #########################################
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    
    return model


# Create checkpoint to save best model for version 3
filepath = './xception_v3_{epoch:02d}_{val_accuracy:.3f}.h5'
checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                             save_best_only=True,
                                             monitor='val_accuracy',
                                             mode='max')

# Set the best values of learning rate and inner layer size based on previous experiments
learning_rate = 0.001
size = 100

# Dict to store results
scores = {}

# List of dropout rates
droprates = [0.0, 0.2, 0.5, 0.8]

for droprate in droprates:
    print(droprate)
    
    model = make_model(learning_rate=learning_rate,
                       size_inner=size,
                       droprate=droprate)
    
    # Train for longer (epochs=30) cause of dropout regularization
    history = model.fit(train_ds, epochs=30, validation_data=val_ds, callbacks=[checkpoint])
    scores[droprate] = history.history
    
    print()
    print()
```
* **Note**: Because we introduce dropout in the neural networks, we will need to train our model for longer, hence, number of `epochs` is set to `30`.
* Also take note that when you analyse the training and validation accuracy plots, the numbers will oscillate due to dropout being added. A single epoch with a high accuracy may only be due to chance, hence to choose the best hyperparameter, consider the converged values instead and the difference between training and validation accuracies. 
* Classes, functions, attributes:
    * `tf.keras.layers.Dropout()`: dropout layer to randomly sets input units (i.e, nodes) to 0 with a frequency of rate at each epoch during training.
    * `rate`: argument to set the fraction of the input units to drop, it is a value of float between 0 and 1.

### 8.10 Data Augmentation
* Other than using `Dropout` to increase model robustness, we could also add more data to our training data.
* Data augmentation is a process of artifically increasing the amount of data by generating new images from existing images. This includes adding minor alterations to images by flipping, rotating (positive value = clockwise direction), shifting (positive value = shift upwards), cropping, zoom-in/out, shearing (pull up/down a corner of image), adding brightness and/or contrast, add black patches, and many more.
* You could perform one or more augmentations on an image.
* Keras `ImageDataGenerator` class has many parameters for data augmentation that we can use for generating data. Important thing to remember that the **data augmentation should only be implemented on train data**, not the validation, because validation acts as unseen data and we do not transform unseen data used for predictions in reality! 
* Here's how we can generate augmented data for training the model:
```python
# Create image generator for train data and also augment the images
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input,
                               rotation_range=30, # implies [-30, 30] range
                               width_shift_range=10.0, # implies [-10, 10] range
                               height_shift_range=10.0, # implies [-10, 10] range
                               shear_range=10,
                               zoom_range=0.1,
                               vertical_flip=True)

train_ds = train_gen.flow_from_directory(directory=train_imgs_dir,
                                         target_size=(150,150),
                                         batch_size=32)
```
* **Note**: `val_ds` in the notebook only has `ImageDataGenerator(preprocessing_function=preprocess_input)` and no transformations.
* **How to choose augmentations**?
    * **Use our own judgement**: E.g. looking at the images (both on train and validation), does it make sense to introduce horizontal flip? It is so for clothing but not so for recognising buildings!
    * **Look at the dataset**, what kind of vairations are there? are objects always center? If so, would it benefit to add rotated or cropped images?
    * **Augmentations are hyperparameters**: like many other hyperparameters, often times we need to test whether image augmentations are useful for the model or not. Say we add in rotation transformation and train the model, **if the model doesn't improve or have same performance after certain epochs (let's say 20), in that case we don't use it**.
* **Note**: Usually augmented data required training for longer.

### 8.11 Training a Larger Model
* Using a smaller model (image size is only (150 x 150)) speeds up the training process where the focus is on hyperparameter-tuning and increasing the robustness of the model.
* Once we're happy with the model's performance, we could increase the image size to (299 x 299) to train a larger model.
* Here we also:
    * Reduced the amount of data augmentation parameters, 
    * Lowered the learning rate and increased the `epochs`. 
    * Added checkpoints
* **Note**: Use a `divide-and-conquer` approach and add in hyperparameters one-by-one instead of using multiple at one go. This helps to isolate those hyperparameters that do not improve performance and exclude them from the final model.
* **Note**: Save the `checkpoints` as different **versions** to associate the versions with the hyperparameters used at a specific version. 
* **Note**: As always, pick the model that gives a great validation accuracy plus the smallest difference between training and validation accuracy.
* This gives us the best results than any previous experiments.

### 8.12 Using the Model
* Earlier we used **h5 format** to save our model when creating the checkpoint. The **HDF5 format** contains the model's architecture, weights values, and `compile()` information. The saved model can be loaded and used for prediction with `keras.models.load_model(path/to/saved_model)` method.
* To evaluate the model and make prediction on test data, we'll need to create the same preprocessing steps for the image as we have done with train and validation data:
```python
# Create image generator for test data
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Path of test images directory
test_imgs_dir = '../input/mlzoomcampimageclassification/zoomcamp-image-classification/clothing-dataset-small/test'

# Load in test images to generator
test_ds = test_gen.flow_from_directory(directory=test_imgs_dir,
                                       target_size=(299,299),
                                       batch_size=32,
                                       shuffle=False)

# Path of an image to make predictions
img_path = 'path/to/image'
# Load image
img = load_img(img_path, target_size=(299,299))

# Convert image to numpy array
x = np.array(img)
# Add batch dimension to the image
X = np.array([x])
# Preprocess the image 
X = preprocess_input(X)
```
* The model performance can be evaluated on test data with `model.evaluate(test_ds)` and the prediction on the test image can be made using the method `model.predict(X)`. We can then `zip` the class names and prediction to see the likelihood.
* The likelihood tells us how likely the image is predicted to be in a particular class. Note that we did not use `softmax` hence the predictions are logits instead of probabilities. The largest logit represents the **greatest likelihood** of this image belonging to this particular class.
* Classes, functions, attributes:
    * `keras.models.load_model()`: method to load saved model
    * `model.evaluate()`: method to evaluate the performance of the model based on the evaluation metrics
    * `model.predict()`: method to make predictions of output depending on the input

### 8.13 Summary
* We can use **pre-trained models** for general image classification.
* **Convolutional layers** let us turn an image into a vector.
* **Dense** layers use the vector to make the predictions.
* Instead of training a model from scratch, we can use **transfer learning** and re-use already trained convolutional layers.
* First, train a small model (150x150) before training a big one (299x299).
* **Learning rate** - how fast the model trains. Fast learner aren't always best ones.
* We can save the best model using **callbacks** and **checkpointing**.
* To **avoid overfitting**, use **dropout** and **augmentation**.

### Homework
* Module 8 [homework questions](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/cohorts/2024/08-deep-learning/homework.md)

### Community Notes
* [Kwang Yang's Notes on PyTorch](https://github.com/kwangyy/ml-zoomcamp/blob/main/Week%208%20Code/Week%208%20-%20Deep%20Learning%20but%20PyTorch.ipynb).
* [Gradio demonstration with Huggingface Space](https://github.com/yusyel/guides/tree/master/huggingface#gradio-demonstration-with-huggingface-space).