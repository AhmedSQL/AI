#!/usr/bin/env python
# coding: utf-8

# In[5]:


import zipfile
zip_ref = zipfile.ZipFile('C://Users//musta//Downloads//archive_141.zip')
zip_ref.extractall('/content')
zip_ref.close()


# In[6]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import losses
import time


# In[7]:


import zipfile
import os

zip_path = 'C://Users//musta//Downloads//archive_141.zip'
extract_dir = 'C://Users//musta//Downloads//extracted_data'

# Extract the ZIP file to the specified directory
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Update the data directory path to the extracted directory
data_dir = os.path.join(extract_dir, 'Skin_Data')

# Rest of your code
batch_size = 32
img_height = 180
img_width = 180

# Continue with the rest of your code, using the updated data_dir variable


# In[8]:


train_dir = os.path.join(data_dir, "Training")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory = r'C:\Users\musta\Downloads\extracted_data\Skin_Data\training',
    labels='inferred',
    label_mode='binary',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)


# In[9]:


val_dir = os.path.join(data_dir, "Training")
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=r'C:\Users\musta\Downloads\extracted_data\Skin_Data\testing',
    labels='inferred',
    label_mode='binary',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


# In[10]:


class_names = train_ds.class_names
print(class_names)


# In[11]:


print(train_ds)


# In[13]:


for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


# In[14]:


normalization_layer = tf.keras.layers.Rescaling(1./255)


# In[15]:


normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))


# In[16]:


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# In[18]:


input_shape = (180,180,3)

model = tf.keras.models.Sequential([
    # since Conv2D is the first layer of the neural network, we should also specify the size of the input
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
    # apply pooling
    tf.keras.layers.MaxPooling2D(2,2),
    # and repeat the process
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    # flatten the result to feed it to the dense layer
    tf.keras.layers.Flatten(), 
    # and define 512 neurons for processing the output coming by the previous layers
    tf.keras.layers.Dense(512, activation='relu'), 
    # a single output neuron. The result will be 0 if the image is a cat, 1 if it is a dog
    tf.keras.layers.Dense(1, activation='sigmoid')  
])
model.summary()


# In[19]:


model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])


# In[20]:


epochs = 20
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=epochs)


# In[21]:


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()


# In[22]:


plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()


# In[30]:


import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam  # Example optimizer, you can choose a different one

# Load the test image
test_img = cv2.imread(r'C:\Users\musta\Downloads\extracted_data\Skin_Data\testing\0\299.JPG')

plt.imshow(test_img)
plt.show()

test_img = cv2.resize(test_img, (256, 256))

test_input = test_img.reshape((1, 256, 256, 3))

# Create the model
model = Sequential()
# Add layers and configure the model architecture

# Compile the model with an optimizer
optimizer = Adam()  # Example optimizer, you can customize it
model.compile(optimizer=optimizer, loss='binary_crossentropy')  # Specify the loss function as well

# Perform prediction
predictions = model.predict(test_input)
print(predictions)


test_img = cv2.imread(r'C:\Users\musta\Downloads\extracted_data\Skin_Data\testing\0\299.JPG')

plt.imshow(test_img)

test_img.shape

test_img = cv2.resize(test_img,(256,256))  # trained data size is 256 * 256

test_input = test_img.reshape((1,256,256,3))  #   1 image of size 256 * 256 and color image with 3 channels

model.predict(test_input)  # 0 refering to cat and 1 refering to dog

test_img = cv2.imread(r'C:\Users\musta\Downloads\extracted_data\Skin_Data\testing\1\5-01.JPG')

test_img = cv2.resize(test_img,(256,256))  # trained data size is 256 * 256

test_input = test_img.reshape((1,256,256,3))  #   1 image of size 256 * 256 and color image with 3 channels

model.predict(test_input)  # 0 refering to cat and 1 refering to dog


# In[ ]:




