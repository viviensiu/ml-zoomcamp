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

