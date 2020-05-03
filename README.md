
my_first_dogvision_project.ipynb
my_first_dogvision_project.ipynb_
This is my First Colab Notebook doing a project on dogvision
Some resources for machine learning model

https://paperswithcode.com/
https://pytorch.org/hub/
https://www.youtube.com/watch?v=oBklltKXtDE&feature=youtu.be&t=173
https://tfhub.dev/
https://modelzoo.co/
https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c
https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4
https://www.youtube.com/watch?v=R9OHn5ZF4Uo
https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
https://www.tensorflow.org/api_docs/python/tf/data/Dataset#unbatch
Project folder has been set up, using google drive. could have used it via kaggle API
Dog Vision project
This notebook builds an end-to-end multi-class image classifier using TensorFlow 2.0 and TensorFlow Hub.

1. Problem
Identifying the breed of a dog,given an image of a dog.

if I saw a dog, I wll capture an image and do my model find the breed of that dog

This kind of problem is called multi-class image classification. It's multi-class because we're trying to classify mutliple different breeds of dog. If we were only trying to classify dogs versus cats, it would be called binary classification.
2. Data
Data is collected from a Kaggle competion Dog Breed Identification
3. Evaluation
The evaluation is a file with prediction probabilities for each dog breed of each test image.

Submission formate reference

4. Feature
Some information about the data:

We're dealing with multi-class image classification problem (unstructured data) so it's probably best we use deep learning/transfer learning.

There are 120 breeds of dogs (this means there are 120 different classes).So it is multi class (My Heart disease project was binary classification because it has only two classes( disease and not disease. italicized text)

There are around 10,000+ images in the training set (these images have labels).

There are around 10,000+ images in the test set (these images have no labels, because we'll want to predict them).

Notes
Multi-class image classification is an important problem because it's the same kind of technology Tesla uses in their self-driving cars or Airbnb uses in atuomatically adding information to their listings.

Since the most important step in a deep learng problem is getting the data ready (turning it into numbers), that's what we're going to start with.

We're going to go through the following TensorFlow/Deep Learning workflow:

Get data ready (download from Kaggle, store, import).
Prepare the data (preprocessing, the 3 sets, X & y).
Choose and fit/train a model (TensorFlow Hub, tf.keras.applications, TensorBoard, EarlyStopping).
Evaluating a model (making predictions, comparing them with the ground truth labels).
Improve the model through experimentation (start with 1000 images, make sure it works, increase the number of images).
Save, sharing and reloading your model (once you're happy with the results).
For preprocessing our data, we're going to use TensorFlow 2.x. The whole premise here is to get our data into Tensors (arrays of numbers which can be run on GPUs) and then allow a machine learning model to find patterns between them.

For our machine learning model, we're going to be using a pretrained deep learning model from TensorFlow Hub.

The process of using a pretrained model and adapting it to your own problem is called transfer learning. We do this because rather than train our own model from scratch (could be timely and expensive), we leverage the patterns of another model which has been trained to classify images.

[ ]
# Unzip the data uploaded on google drive.
#!unzip 'drive/My Drive/Colab Notebooks/kaggle/dog-breed-identification.zip' -d 'drive/My Drive/Machine_Learning_with_python/Machine Learning Zero to Mastery ZTM/dog_vision'
Getting Our workspace Ready
Import TensorFlow 2.x ✅

Import TensorFlow Hub ✅

TensorFlow Hub is a repository of reusable assets for machine learning with TensorFlow. In particular, it provides pre-trained SavedModels that can be reused to solve new tasks with less training time and less training data. https://github.com/tensorflow/hub
Make sure we're using a GPU ✅

[ ]
# import Tensorflow and Tensosorflow hub to colab

import tensorflow as tf
import tensorflow_hub as hub
print('TF version:', tf.__version__)
print('TF Hub version:', hub.__version__)

# Checking GPU availabiity
print('GPU','available :)' if tf.config.list_physical_devices('GPU') else 'is not available :(' )
[ ]
# if need another tf version
# Run this cell if TensorFlow 2.x isn't the default in Colab
# try:
#   # %tensorflow_version only exists in Colab
#   %tensorflow_version 2.x
# except Exception:
#   pass
Getting our Darta Ready
There are a few ways we could do this. Many of them are detailed in the Google Colab notebook on I/O (input and output).

Like all machine learning project, data needs to be numeric. So I am going to turn images into Tensors.(numerical formate (matrix)

And because the data we're using is hosted on Kaggle, we could even use the Kaggle API.

Let's start by checking the labels

[ ]
# # Running this cell will provide you with a token to link your drive to this notebook
# from google.colab import drive
# drive.mount('/content/drive')
[ ]
import pandas as pd

labels_csv = pd.read_csv('/content/drive/My Drive/Machine_Learning_with_python/Machine Learning Zero to Mastery ZTM/dog_vision/labels.csv')
labels_csv.describe()
[ ]
labels_csv.head()

[ ]
# How many images are there for each breed
labels_csv.breed.value_counts().plot.bar(figsize=(30,15));
[ ]
labels_csv['breed'].value_counts().median()
we need at least 10 imaage ,100+ in recomended. for learnig. see link for details Preparing your training data

in here our data distribution is good.

[ ]
from IPython.display import Image
Image('/content/drive/My Drive/Machine_Learning_with_python/Machine Learning Zero to Mastery ZTM/dog_vision/train/001513dfcb2ffafc82cccf4d8bbaba97.jpg')
[ ]
Getting Images & their labels
Let's get a list of all our image file path name

[ ]
labels_csv.head()
Since we've got the image ID's and their labels in a DataFrame (labels_csv), we'll use it to create:

A list a filepaths to training images
An array of all labels
An array of all unique labels
We'll only create a list of filepaths to images rather than importing them all to begin with. This is because working with filepaths (strings) is much efficient than working with images.

[ ]
# Create pathnames from image ID's
filename = ['/content/drive/My Drive/Machine_Learning_with_python/Machine Learning Zero to Mastery ZTM/dog_vision/train/'  + filename + '.jpg' for filename in labels_csv['id']]



filename[10220:]  # kaggle said it has 10222 files. I just checked that
Now we've got a list of all the filenames from the ID column of labels_csv, we can compare it to the number of files in our training data directory to see if they line up.

If they do, great. If not, there may have been an issue when unzipping the data (what we did above), to fix this, you might have to unzip the data again. Be careful not to let your Colab notebook disconnect whilst unzipping.

[ ]
import os
if len(os.listdir('/content/drive/My Drive/Machine_Learning_with_python/Machine Learning Zero to Mastery ZTM/dog_vision/train/')) == len(filename):
  print(f'filename matches the actual amount {len(filename)} as per kaggle')
else:  
  print(f'filename shortage than actual amount as per kaggle')


Note: Python - OS Module: It is possible to automatically perform many operating system tasks. The OS module in Python provides functions for creating and removing a directory (folder), fetching its contents, changing and identifying the current directory, etc.

The listdir() function returns the list of all files and directories in the specified directory. https://www.tutorialsteacher.com/python/os-module
[ ]
os.listdir('/content/drive/My Drive/Machine_Learning_with_python/Machine Learning Zero to Mastery ZTM/dog_vision/train/')[:10]
[ ]
# Check an image directly from a filepath
print(labels_csv.breed[10221])
Image(filename[10221]) # the last image
Preparing Labels
since we've now got our training image filepath in a list, let's prepare our labels We'll take them from labels_csv and turn them into a NumPy array.

[ ]
labels = labels_csv.breed
labels
[ ]
# Converting it to number
import numpy as np
# labels = np.array(labels)
#or we can do it like this
labels = labels_csv['breed'].to_numpy()
labels
[ ]
# Cheking for missing data
if len(labels) == len(filename):
  print(f'Number of filenamepath matches with labels {len(labels)}. We can proceed')
else:
  print(f'Nnmber of labels {len(labels)} and filename {len(filename)} does not matces')
[ ]
labels.dtype # those are string object

[ ]
# finding unique label value
unique_breed = np.unique(labels)
print(unique_breed)
len(unique_breed)  # 120 types breed as per kaggle
Why do it like this?

Remember, an important concept in machine learning is converting your data to numbers before passing it to a machine learning model.

In this case, we've transformed a single dog breed name such as boston_bull into a one-hot array.

[ ]
print(labels[0])
unique_breed == labels[0]
from 10000+ labels we created 10000+ arrays.
each array has one value 1(TRUE) but at different position
and there are 120 different position (unique_breed)
that means 120 class for multiclass classification
[ ]
# Turn every labels in boolean array
boolean_labels = [ i == unique_breed for i in labels]
print(len(boolean_labels))
boolean_labels[:1]
[ ]
## now example of turning boolean array into integers

print(labels[0]) # original label
print(np.where(unique_breed == labels[0])) # index where lable[0] occured
print(boolean_labels[0].argmax()) # where label occued in booleanarray
print(boolean_labels[0].astype(int))
unique_breed[19]

Note*

We made a array from 10k+ labels & then find the unique type 120 type in unique_breed.
after comapring 10k+ label with 120 type we get 10k+ array where one 1 is present in each array
position of this 1 represents the type in 120 types
boolean_labels[0].argmax() is 19
unique_breed 19 index named the type of dog
[ ]
boolean_labels[:2]
[ ]
filename[:2]
Creating our Valdation set
Set up X & y
[ ]
X = filename
y = boolean_labels
We are going to start our experiment with 1000 sample to saave time

Checkout the magic function forms like the slider here https://colab.research.google.com/notebooks/forms.ipynb
[ ]
# set number of images for experimenting
NUM_IMAGES = 1000 # @param {type:'slider', min:1000, max:10000,step:1000}
NUM_IMAGES
NUM_IMAGES:

1000
Spliting data
[ ]
# Let's Split the data
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X[:NUM_IMAGES],
                                                 y[:NUM_IMAGES],
                                                 test_size=0.2,
                                                 random_state = 42)   # we may also use tarin_size
len(X_train),len(X_val), len(y_train),len(y_val)
[ ]
X_train[:2], y_train[:2]
Preprocessing Data ( Turning images into Tensors)
Our labels are in numeric format but our images are still just file paths.

Since we're using TensorFlow, our data has to be in the form of Tensors.

A Tensor is a way to represent information in numbers. If you're familar with NumPy arrays (you should be), a Tensor can be thought of as a combination of NumPy arrays, except with the special ability to be used on a GPU.

Because of how TensorFlow stores information (in Tensors), it allows machine learning and deep learning models to be run on GPUs (generally faster at numerical computing).

To preprocess our data we will write a function which will do the below tasks:

Take an image filepath as input
Use tensorflow to read the file and save it to a variable, 'image'
turn our 'image' (a jpg) into Tensors
Converting colour chanel value from 0-255 to 0-1 called normalization
Resize the image to a shape of (224,224)
return the modified image
Reference : 1. https://www.tensorflow.org/tutorials/load_data/images) 2. https://www.tensorflow.org/guide/data

Before we do, Let's see what importing an image looks like

[ ]
from matplotlib.pyplot import imread
image = imread(filename[42])
image.shape

[ ]
image[:2]
[ ]
image.max(), image.min() # due to RGB colur chanel max is 255 and min is 0
[ ]
# converting to tensors
tf.constant(image)[:2]
# same like array but it is tensor and can be processed in GPU
Function to process image
[ ]
# Define image size
IMG_SIZE = 224

#Creating funtion
def process_image(image_path, img_size = IMG_SIZE):
  '''
  Take an image file path and turns image into tensors and resize it.
  '''
  # Read image file
  image = tf.io.read_file(image_path)

Turning Data into batches
Wonderful. Now we've got a function to convert our images into Tensors, we'll now build one to turn our data into batches (more specifically, a TensorFlow BatchDataset).

What's a batch?

A batch (also called mini-batch) is a small portion of your data, say 32 (32 is generally the default batch size) images and their labels. In deep learning, instead of finding patterns in an entire dataset at the same time, you often find them one batch at a time.

Let's say you're dealing with 10,000+ images (which we are). Together, these files may take up more memory than your GPU has. Trying to compute on them all would result in an error.

Instead, it's more efficient to create smaller batches of your data and compute on one batch at a time.

TensorFlow is very efficient when your data is in batches of (image, label) Tensors. So we'll build a function to do create those first. We'll take advantage of of process_image function at the same time. Tf documentation says it is goog to use batch. should use batch size of 32

Why turning Data into batches??

Let's say we are trying to process 10000+ images at one go... they might not fit into model

So that's why, we do about 32 images at time called batch size. manual adjustment of batch size can be done if needed

in order to use tf efffectively, we need our data in the form of tensor tuples looks like this: (image,label)

Functoin to return a tuples
[ ]
# Create a simple function to return a tuple (image, label)
def get_image_label(image_path,label):

  """
  Takes an image file path name and the associated label,
  processes the image and returns a tuple of (image, label).
  """
  image = process_image(image_path)

  return image,label
[ ]
process_image(X[42])
[ ]
get_image_label(X[42], tf.constant(y[42]))
Function to create Batches
Now we got a way to turn our data into tuples of Tensors in the form (image,label), lets make a funtion to turn all of our data (X & y) into batches

Because we'll be dealing with 3 different sets of data (training, validation and test), we'll make sure the function can accomodate for each set.
We'll set a default batch size of 32 because according to Yann Lecun (one of the OG's of deep learning), friends don't let friends train with batch sizes over 32.

[ ]
# Create a Function to write data into batches
BATCH_SIZE = 32

def create_data_batches(X, y=None, batch_size=BATCH_SIZE,valid_data=False, test_data=False ):
  '''
  Create a batch of data out of image (X) and  label (y) pairs.
  Shuffles the data if it is training data but does not shuffles for validation data.
  Also accept test data as input.
  '''
  # if the data is test data, we may not have any labels

[ ]
train_data = create_data_batches(X_train,y_train)
valid_data = create_data_batches(X_val,y_val,valid_data=True)
[ ]
# different atributes of our databatces
train_data.element_spec, valid_data.element_spec
Look at that! We've got our data in batches, more specifically, they're in Tensor pairs of (images, labels) ready for use on a GPU.

But having our data in batches can be a bit of a hard concept to understand. Let's build a function which helps us visualize what's going on under the hood.

Visulization
[ ]
# create a function for visualization image
import matplotlib.pyplot as plt
import numpy as np


def show_25_image(images,labels):
    plt.figure(figsize=(10,10))
    for i in range(25):
        ax = plt.subplot(5, 5, i+1)
        plt.imshow(images[i])

To make computation efficient, a batch is a tighly wound collection of Tensors.

So to view data in a batch, we've got to unwind it.

We can do so by calling the as_numpy_iterator() method on a data batch.

This will turn our a data batch into something which can be iterated over.

Passing an iterable to next() will return the next item in the iterator.

In our case, next will return a batch of 32 images and label pairs.

Note: Running the cell below and loading images may take a little while.

[ ]
train_image, train_label = next(train_data.as_numpy_iterator())
type(train_image)
show_25_image(train_image,train_label)

Note: validation showing same image because. we shuffled train dataset. test and validation were not shuffled

[ ]
valid_image, valid_label = next(valid_data.as_numpy_iterator())
show_25_image(valid_image,valid_label)
Creating and training a model
Now our data is ready, let's prepare it modelling. We'll use an existing model from TensorFlow Hub.

TensorFlow Hub is a resource where you can find pretrained machine learning models for the problem you're working on.

Using a pretrained machine learning model is often referred to as transfer learning.

Why use a pretrained model?
Building a machine learning model and training it on lots from scratch can be expensive and time consuming.

Transfer learning helps eliviate some of these by taking what another model has learned and using that information with your own problem.

How do we choose a model?
Since we know our problem is image classification (classifying different dog breeds), we can navigate the TensorFlow Hub page by our problem domain (image).

We start by choosing the image problem domain, and then can filter it down by subdomains, in our case, image classification.

Doing this gives a list of different pretrained models we can apply to our task.

Clicking on one gives us information about the model as well as instructions for using it.

For example, clicking on the mobilenet_v2_130_224 model, tells us this model takes an input of images in the shape 224, 224. It also says the model has been trained in the domain of image classification.

Let's try it out.

Building Model
before start few things need to define:

input shape(our images shape,in the form of Tensors) to our model
output shape(our images label,in the form of Tensors) to our model
URL of the model we want to use https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4
[ ]
# Setup input shape to our model
INPUT_SHAPE = [None, IMG_SIZE,IMG_SIZE,3] # [batch, height,width, colour channel ]

#Setup Output shape of model
OUTPUT_SHAPE = len(unique_breed)

# Setup URL from Tensorflow Hub
MODEL_URL = 'https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4'
Keras Deep learning model
Now we have input,output and model url, Let's put it together into a keras deep learning model

Let's create a function which:

takes the input shape, output shape and the model we have choosen as parameters

Define the layers in a Keras model in sequential fashion ( do this first, then this and then this type)

compiles the model (says it should be evaluated and improved)

Builds the model (tells the model the input shape it's going to take).

Return the model

All of this can be found here https://www.tensorflow.org/guide/keras/overview
[ ]
# Create a  function to c build keras model

def create_model(input_shape=INPUT_SHAPE,output_shape=OUTPUT_SHAPE,model_url= MODEL_URL):
  print('Building model with:', MODEL_URL)

  # setup model Layers
  model = tf.keras.Sequential([hub.KerasLayer(MODEL_URL), #Layer 1 (input layer)
                               tf.keras.layers.Dense(units=OUTPUT_SHAPE,
                                                     activation='softmax') # Layer 2 ( output layer)                               
            
What's happening here?

Setting up the model layers
There are two ways to do this in Keras, the functional and sequential API. We've used the sequential.

Which one should you use?

The Keras documentation states the functional API is the way to go for defining complex models but the sequential API (a linear stack of layers) is perfectly fine for getting started, which is what we're doing.

The first layer we use is the model from TensorFlow Hub (hub.KerasLayer(MODEL_URL). So our first layer is actually an entire model (many more layers). This input layer takes in our images and finds patterns in them based on the patterns mobilenet_v2_130_224 has found.

The next layer (tf.keras.layers.Dense()) is the output layer of our model. It brings all of the information discovered in the input layer together and outputs it in the shape we're after, 120 (the number of unique labels we have)

The activation="softmax" parameter tells the output layer, we'd like to assign a probability value to each of the 120 labels somewhere between 0 & 1. The higher the value, the more the model believes the input image should have that label. If we were working on a binary classification problem, we'd use activation="sigmoid".

For more on which activation function to use, see the article Which Loss and Activation Functions Should I Use?

Compiling the model
This one is best explained with a story.

Let's say you're at the international hill descending championships. Where your start standing on top of a hill and your goal is to get to the bottom of the hill. The catch is you're blindfolded.

Luckily, your friend Adam is standing at the bottom of the hill shouting instructions on how to get down.

At the bottom of the hill there's a judge evaluating how you're doing. They know where you need to end up so they compare how you're doing to where you're supposed to be. Their comparison is how you get scored.

Transferring this to model.compile() terminology:

loss - The height of the hill is the loss function, the models goal is to minimize this, getting to 0 (the bottom of the hill) means the model is learning perfectly.
optimizer - Your friend Adam is the optimizer, he's the one telling you how to navigate the hill (lower the loss function) based on what you've done so far. His name is Adam because the Adam optimizer is a great general which performs well on most models. Other optimizers include RMSprop and Stochastic Gradient Descent.
metrics - This is the onlooker at the bottom of the hill rating how well your perfomance is. Or in our case, giving the accuracy of how well our model is predicting the correct image label.
Building the model
We use model.build() whenever we're using a layer from TensorFlow Hub to tell our model what input shape it can expect.

In this case, the input shape is [None, IMG_SIZE, IMG_SIZE, 3] or [None, 224, 224, 3] or [batch_size, img_height, img_width, color_channels].

Batch size is left as None as this is inferred from the data we pass the model. In our case, it'll be 32 since that's what we've set up our data batches as.

Now we've gone through each section of the function, let's use it to create a model.

We can call summary() on our model to get idea of what our model looks like.

[ ]
 model = create_model()
 model.summary()

The non-trainable parameters are the patterns learned by mobilenet_v2_130_224 and the trainable parameters are the ones in the dense layer we added.

This means the main bulk of the information in our model has already been learned and we're going to take that and adapt it to our own problem.

Creating callbacks
We've got a model ready to go but before we train it we'll make some callbacks.

Callbacks are helper functions a model can use during training to do things such as save a models progress, check a models progress or stop training early if a model stops improving.

The two callbacks we're going to add are a TensorBoard callback and an Early Stopping callback.

TensorBoard Callback
TensorBoard helps provide a visual way to monitor the progress of your model during and after training.

It can be used directly in a notebook to track the performance measures of a model such as loss and accuracy.

To set up a TensorBoard callback and view TensorBoard in a notebook, we need to do three things:

Load the TensorBoard notebook extension.
Create a TensorBoard callback which is able to save logs to a directory and pass it to our model's fit() function.
Visualize the our models training logs using the %tensorboard magic function (we'll do this later on).
TensorBoard Callback
[ ]
# Load TensorBoard Notebook Extension
%load_ext tensorboard
[ ]
import datetime

# Create a function to build TensorBoard calback
def create_tensorboard_callback():
  # create a log directory to save tensorboard log
  log_dir = os.path.join('/content/drive/My Drive/Machine_Learning_with_python/Machine Learning Zero to Mastery ZTM/dog_vision/logs',
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
  return tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  

Early Stoping callback
Early stopping helps prevent overfitting by stopping a model when a certain evaluation metric stops improving. If a model trains for too long, it can do so well at finding patterns in a certain dataset that it's not able to use those patterns on another dataset it hasn't seen before (doesn't generalize).

It's basically like saying to our model, "keep finding patterns until the quality of those patterns starts to go down."

[ ]
# Create Early stopping callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=3)
Training a model (on a subset of data)
Our first model is only going to be trained on 1000 images. Or trained on 800 images and then validated on 200 images, meaning 1000 images total or about 10% of the total data.

We do this to make sure everything is working. And if it is, we can step it up later and train on the entire training dataset.

The final parameter we'll define before training is NUM_EPOCHS (also known as number of epochs).

NUM_EPOCHS defines how many passes of the data we'd like our model to do. A pass is equivalent to our model trying to find patterns in each dog image and see which patterns relate to each label.

If NUM_EPOCHS=1, the model will only look at the data once and will probably score badly because it hasn't a chance to correct itself. It would be like you competing in the international hill descent championships and your friend Adam only being able to give you 1 single instruction to get down the hill.

What's a good value for NUM_EPOCHS?

This one is hard to say. 10 could be a good start but so could 100. This is one of the reasons we created an early stopping callback. Having early stopping setup means if we set NUM_EPOCHS to 100 but our model stops improving after 22 epochs, it'll stop training.

Along with this, let's quickly check if we're still using a GPU.

[ ]
NUM_EPOCH = 100 # @param {type:'slider', min:10, max:100, step:10}
NUM_EPOCH:

100
[ ]
# Check to make sure we are still using GPU
print('GPU is available' if tf.config.list_physical_devices('GPU') else 'GPU is not available')
Boom! We've got a GPU running and NUM_EPOCHS setup. Let's create a simple function which trains a model. The function will:

Create a model using create_model().
Setup a TensorBoard callback using create_tensorboard_callback() (we do this here so it creates a log directory of the current date and time).
Call the fit() function on our model passing it the training data, validatation data, number of epochs to train for and the callbacks we'd like to use.
Return the fitted model.
Function to train model
[ ]
train_data, valid_data
[ ]
# Build a fuction to train and return a trained model
def train_model():
  '''
  Trains a given model and returns the trained version
  '''
  # Create model
  model = create_model()

  # create TensorBoard session everytime we trained a model
  tensorboard = create_tensorboard_callback()

Note: When training a model for the first time, the first epoch will take a while to load compared to the rest. This is because the model is getting ready and the data is being initialised. Using more data will generally take longer, which is why we've started with ~1000 images. After the first epoch, subsequent epochs should take a few second

[ ]
# fit the model to the data
# model = train_model()
Question: It looks like our model might be overfitting (getting far better results on the training set than the validation set), what are some ways to prevent model overfitting? Hint: this may involve searching something like "ways to prevent overfitting in a deep learning model?".

Note: Overfitting to begin with is a good thing. It means our model is learning something.

Checking Tensorboard logs
Now our model has been trained, we can make its performance visual by checking the TensorBoard logs.

The TensorBoard magic function (%tensorboard) will access the logs directory we created earlier and viualize its contents.

[ ]
%tensorboard --logdir /content/drive/My\ Drive/Machine_Learning_with_python/Machine\ Learning\ Zero\ to\ Mastery\ ZTM/dog_vision/logs
Thanks to our early_stopping callback, the model stopped training after 26 or so epochs (in my case, yours might be slightly different). This is because the validation accuracy failed to improve for 3 epochs.

But the good new is, we can definitely see our model is learning something. The validation accuracy got to 65% in only a few minutes.

This means, if we were to scale up the number of images, hopefully we'd see the accuracy increase.

Making and evaluating predictions using a trained model
Before we scale up and train on more data, let's see some other ways we can evaluate our model. Because although accuracy is a pretty good indicator of how our model is doing, it would be even better if we could could see it in action.

Making predictions with a trained model is as calling predict() on it and passing it data in the same format the model was trained on.

[ ]
valid_data
[ ]
predictions = model.predict(valid_data,verbose=1)
predictions
[ ]
predictions[0]
NOte: there are 120 probabilities for a single image, highest will be the label

now yo know what is softmax doing

[ ]
 predictions.shape
[ ]
np.sum(predictions[0])
[ ]
index= 0
print(predictions[index])
print(np.sum(predictions[index]))
print(np.argmax(predictions[index]))
print(np.max(predictions[index]))
print(unique_breed[np.argmax(predictions[index])])


[ ]
unique_breed[17]
Making predictions with our model returns an array with a different value for each label.

In this case, making predictions on the validation data (200 images) returns an array (predictions) of arrays, each containing 120 different values (one for each unique dog breed).

These different values are the probabilities or the likelihood the model has predicted a certain image being a certain breed of dog. The higher the value, the more likely the model thinks a given image is a specific breed of dog.

Let's see how we'd convert an array of probabilities into an actual label.

Function to get predicted labals
[ ]
def get_pred_label(predictions):
  '''
  turn prediction probabilities into labels
  '''
  return unique_breed[np.argmax(predictions)]
[ ]
get_pred_label(predictions[0])
Wonderful! Now we've got a list of all different predictions our model has made, we'll do the same for the validation images and validation labels.

Remember, the model hasn't trained on the validation data, during the fit() function, it only used the validation data to evaluate itself. So we can use the validation images to visually compare our models predictions with the validation labels.

Since our validation data (val_data) is in batch form, to get a list of validation images and labels, we'll have to unbatch it (using unbatch()) and then turn it into an iterator using as_numpy_iterator().

Let's make a small function to do so.

[ ]
# let's unbatch the data
image_ = []
label_ = []
for image,label in valid_data.unbatch().as_numpy_iterator():
  image_.append(image)
  label_.append(label)


[ ]
get_pred_label(label_[0])
model got right prediction at 0

[ ]
print(get_pred_label(predictions[30]))
get_pred_label(label_[30])
this time model predicts wrong. now lets create the fnction

Function to unbatch
[ ]
# creating function to unbatchify this
def unbatch(unbatch_data):
  '''
  will unbatch the (image,label) touple of Tensors to separate list
  '''
  image = []
  label = []

  for img, lbl in unbatch_data.unbatch().as_numpy_iterator():
    image.append(img)

[ ]
val_image, val_label = unbatch(valid_data)
val_image[0], val_label[0]
Nailed it!

Now we've got ways to get:

Prediction labels
Validation labels (truth labels)
Validation images
Let's make some functions to make these all a bit more visualize.

More specifically, we want to be able to view an image, its predicted label and its actual label (true label).

The first function we'll create will:

Take an array of prediction probabilities, an array of truth labels, an array of images and an integer.
Convert the prediction probabilities to a predicted label.
Plot the predicted label, its predicted probability, the truth label and target image on a single plot.
Function to plot prediction
[ ]
def plot_pred(prediction_probabilities, truth_labels,images,n=1):
  '''
  View the prediction, ground truth and image for sample n
  '''
  pred_prob, true_label, image = prediction_probabilities[n], truth_labels[n], images[n]
  
  # get predicted label
  pred_label = get_pred_label(pred_prob)

  #plot image

[ ]
plot_pred(prediction_probabilities=predictions,truth_labels=val_label,images=val_image,n=20)
Nice! Making functions to help visual your models results are really helpful in understanding how your model is doing.

Since we're working with a multi-class problem (120 different dog breeds), it would also be good to see what other guesses our model is making. More specifically, if our model predicts a certain label with 24% probability, what else did it predict?

Let's build a function to demonstrate. The function will:

Take an input of a prediction probabilities array, a ground truth labels array and an integer.
Find the predicted label using get_pred_label().
Find the top 10:
Prediction probabilities indexes
Prediction probabilities values
Prediction labels
Plot the top 10 prediction probability values and labels, coloring the true label green.
[ ]
a=np.array([55,75,95,15,35,25,45])
a
[ ]
a[-3:]
[ ]
a[-3:][::-1]
[ ]
a.argsort()  # index number from small to big
[ ]
a.argsort()[::-1]# index number from big to to small
Function to plot top 10
[ ]
def plot_pred_conf(prediction_probabilities,truth_label,n=1):
  '''
  Plots the top 10 highest prediction confidences along with
  the truth label for sample n.
  '''
  pred_prob, true_label = prediction_probabilities[n],truth_label[n]

  #Get prediction label
  pred_label = get_pred_label(pred_prob)
  

[ ]
plot_pred_conf(predictions,val_label,n=3)
Wonderful! Now we've got some functions to help us visualize our predictions and evaluate our model, let's check out a few.

[ ]


for i in range(5):
  plt.subplots(figsize=(5,5))
  plot_pred(predictions,
          val_label,
          val_image,n=i+1)
  
  plt.subplots(figsize=(7,3))
  plot_pred_conf(predictions,

Save and load a model
After training a model, it's a good idea to save it. Saving it means you can share it with colleagues, put it in an application and more importantly, won't have to go through the potentially expensive step of retraining it.

The format of an entire saved Keras model is h5. So we'll make a function which can take a model as input and utilise the save() method to save it as a h5 file to a specified directory.

Function to save a model
[ ]
# let's create a function to save model
def save_model(model,suffix=None):
  '''
  Saves a given model in a models directory and appends a suffix (str)
  for clarity and reuse.

  '''

  model_dir = os.path.join('/content/drive/My Drive/Machine_Learning_with_python/Machine Learning Zero to Mastery ZTM/dog_vision/model',
                           datetime.datetime.now().strftime('%Y%m%d-%H%M%s'))

[ ]
# save it
save_model(model,suffix='trained with 1000 images-mobilenetv2-Adam')
If we've got a saved model, we'd like to load it, let's create a function which can take a model path and use the tf.keras.models.load_model() function to load it into the notebook.

Because we're using a component from TensorFlow Hub (hub.KerasLayer) we'll have to pass this as a parameter to the custom_objects parameter.

Function to load a model
[ ]
def load_model(modelpath):
  '''
   Loads a saved model from a specified path.
  '''
  print(f'Loading model from {modelpath}')
  model = tf.keras.models.load_model(modelpath,
                                     custom_objects={"KerasLayer":hub.KerasLayer}
                                     )
  return model


[ ]
load_1000_trained_model = load_model('/content/drive/My Drive/Machine_Learning_with_python/Machine Learning Zero to Mastery ZTM/dog_vision/model/20200501-18331588358002-trained with 1000 images-mobilenetv2-Adam.h5')
let's check if it works

[ ]
valid_data
[ ]
model.evaluate(valid_data)
[ ]
load_1000_trained_model.evaluate(valid_data)
Traing model on Full Dataset
Now we know our model works on a subset of the data, we can start to move forward with training one on the full data.

Above, we saved all of the training filepaths to X and all of the training labels to y. Let's check them out.

[ ]
len(X), len(y)
There we go! We've got over 10,000 images and labels in our training set.

Before we can train a model on these, we'll have to turn them into a data batch.

The beautiful thing is, we can use our create_data_batches() function from above which also preprocesses our images for us (thank you past us for writing a helpful function).

[ ]
# Create databatches on full set
full_data_batch = create_data_batches(X,y)
[ ]
full_data_batch.element_spec
Our data is in a data batch, all we need now is a model.

And surprise, we've got a function for that too! Let's use create_model() to instantiate another model.

[ ]
# Create model on full data
full_model = create_model()
[ ]
# full model callbacks
full_model_tensorboard = create_tensorboard_callback()

# NO validation set when training on all data, So we can't monitor validation accuracy
full_model_early_stopping = tf.keras.callbacks.EarlyStopping('accuracy',
                                                             patience = 3)
Note: Since running the cell below will cause the model to train on all of the data (10,000+) images, it may take a fairly long time to get started and finish. However, thanks to our full_model_early_stopping callback, it'll stop before it starts going too long.

Remember, the first epoch is always the longest as data gets loaded into memory. After it's there, it'll speed up.

[ ]
# full_model.fit(x=full_data_batch,
#                epochs=100,
#                callbacks=[full_model_tensorboard,full_model_early_stopping])
[ ]
# save_model(full_model,suffix='trained with full images-mobilenetv2-Adam')
[ ]
%tensorboard --logdir /content/drive/My\ Drive/Machine_Learning_with_python/Machine\ Learning\ Zero\ to\ Mastery\ ZTM/dog_vision/logs
Loading full model
[ ]
load_full_trained_model = load_model('/content/drive/My Drive/Machine_Learning_with_python/Machine Learning Zero to Mastery ZTM/dog_vision/model/20200501-21501588369839-trained with full images-mobilenetv2-Adam.h5')
Making predictions on the test dataset
Since our model has been trained on images in the form of Tensor batches, to make predictions on the test data, we'll have to get it into the same format.

Luckily we created create_data_batches() earlier which can take a list of filenames as input and convert them into Tensor batches.

To make predictions on the test data, we'll:

Get the test image filenames.
Convert the filenames into test data batches using create_data_batches() and setting the test_data parameter to True (since there are no labels with the test images).
Make a predictions array by passing the test data batches to the predict() function.
[ ]
import os
test_data = os.listdir('/content/drive/My Drive/Machine_Learning_with_python/Machine Learning Zero to Mastery ZTM/dog_vision/test')
[ ]
test_data[:10]
[ ]
# creating test data path
test_data_path = ['/content/drive/My Drive/Machine_Learning_with_python/Machine Learning Zero to Mastery ZTM/dog_vision/test/' + filename for filename in test_data]
[ ]
len(test_data_path)
[ ]
# creating test data batch

test_data_batch = create_data_batches(X=test_data_path,test_data=True)
[ ]
test_data_batch
[ ]
# Predicting on test data batch
test_data_prediction = load_full_trained_model.predict(test_data_batch,verbose=1)
[ ]
import numpy as np
import pandas as pd


[ ]
#saving it to csv to nut making prediction again. it took 1h for me
np.savetxt('/content/drive/My Drive/Machine_Learning_with_python/Machine Learning Zero to Mastery ZTM/dog_vision/model_preds/preds_array.csv',test_data_prediction,delimiter=',')
[ ]

pred_df = pd.read_csv('/content/drive/My Drive/Machine_Learning_with_python/Machine Learning Zero to Mastery ZTM/dog_vision/model_preds/preds_array.csv')
[ ]
pred_df
[ ]
prediction_dataframe = pd.DataFrame(columns=['id']+ list(unique_breed))
[ ]
prediction_dataframe
[ ]
Mounting Google Drive...
