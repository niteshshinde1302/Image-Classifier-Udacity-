import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import json
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
dataset, dataset_info = tfds.load ("oxford_flowers102", as_supervised= True, with_info = True)
# TODO: Create a training set, a validation set and a test set.
train_set = dataset['train']
test_set = dataset['test']
val_set = dataset['validation']

# TODO: Get the number of examples in each set from the dataset info.
num_training_examples = dataset_info.splits['train'].num_examples
print('There are {:,} images in the training set'.format(num_training_examples))
num_testing_examples = dataset_info.splits['test'].num_examples
print('There are {:,} images in the testing set'.format(num_testing_examples))
num_validation_examples = dataset_info.splits['validation'].num_examples
print('There are {:,} images in the validation set'.format(num_validation_examples))
# TODO: Get the number of classes in the dataset from the dataset info.
num_classes = dataset_info.features['label'].num_classes
print('There are {:,} classes in our dataset'.format(num_classes))
total_num_examples = num_training_examples+num_testing_examples+num_validation_examples

with open('label_map.json', 'r') as f:
    class_names = json.load(f)

batch_size = 32
image_size = 224

def format_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image, label

training_batches = train_set.shuffle(num_training_examples//4).map(format_image).batch(batch_size).prefetch(1)
validation_batches = val_set.map(format_image).batch(batch_size).prefetch(1)
testing_batches = test_set.map(format_image).batch(batch_size).prefetch(1)

URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL, input_shape=(image_size, image_size,3))
feature_extractor.trainable = False

model = tf.keras.Sequential([feature_extractor,tf.keras.layers.Dense(num_classes, activation = 'softmax')])

print(model.summary())

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

EPOCHS = 50

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(training_batches,epochs=EPOCHS,validation_data=validation_batches,callbacks=[early_stopping])

# TODO: Print the loss and accuracy values achieved on the entire test set.
loss, accuracy = model.evaluate(testing_batches)

print('\nLoss on the TEST Set: {:,.3f}'.format(loss))
print('Accuracy on the TEST Set: {:.3%}'.format(accuracy))

# TODO: Save your trained model as a Keras model.
saved_keras_model_filepath = 'my_model.h5'
model.save(saved_keras_model_filepath)