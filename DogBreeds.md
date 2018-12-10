---

title: EDA
notebook: DogBreeds.ipynb
nav_include: 2

---



##Dog Breeds Models


```
from google.colab import files
uploaded = files.upload()
```




     <input type="file" id="files-386c0550-58d1-4ad2-a3ae-465e9a400967" name="files[]" multiple disabled />
     <output id="result-386c0550-58d1-4ad2-a3ae-465e9a400967">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script src="/nbextensions/google.colab/files.js"></script> 


    Saving test_list.mat to test_list (1).mat
    Saving train_list.mat to train_list (1).mat




```
from google.colab import files
uploaded = files.upload()
```




     <input type="file" id="files-e4e3440d-edd8-4259-9b03-0e9fa3937ba4" name="files[]" multiple disabled />
     <output id="result-e4e3440d-edd8-4259-9b03-0e9fa3937ba4">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script src="/nbextensions/google.colab/files.js"></script> 


    Saving images.tar to images.tar




```
ls
```


     [0m[01;34mdrive[0m/       [01;34msample_data[0m/         test_list.mat         train_list.mat
     images.tar  'test_list (1).mat'  'train_list (1).mat'




```
 !pip install imageio
```


    Requirement already satisfied: imageio in /usr/local/lib/python3.6/dist-packages (2.4.1)
    Requirement already satisfied: pillow in /usr/local/lib/python3.6/dist-packages (from imageio) (4.0.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from imageio) (1.14.6)
    Requirement already satisfied: olefile in /usr/local/lib/python3.6/dist-packages (from pillow->imageio) (0.46)




```
!tar -xvf images.tar
```




```
#Libraries 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage, io, misc
import numpy as np
import pandas as pd
from matplotlib.pyplot import imshow
%matplotlib inline
import imageio
import os

from tqdm import tqdm

import keras 
from keras.preprocessing import image                  
from keras.callbacks import ModelCheckpoint  
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Activation, Dense, Flatten
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True 

import seaborn as sns

```


    Using TensorFlow backend.




```
# Load the file_list.mat to get the list of all files

train_files = io.loadmat('train_list.mat')['file_list']
train_targets = io.loadmat('train_list.mat')['labels']
display(train_files)
display(train_targets)

```



    array([[array(['n02085620-Chihuahua/n02085620_5927.jpg'], dtype='<U38')],
           [array(['n02085620-Chihuahua/n02085620_4441.jpg'], dtype='<U38')],
           [array(['n02085620-Chihuahua/n02085620_1502.jpg'], dtype='<U38')],
           ...,
           [array(['n02116738-African_hunting_dog/n02116738_6754.jpg'], dtype='<U48')],
           [array(['n02116738-African_hunting_dog/n02116738_9333.jpg'], dtype='<U48')],
           [array(['n02116738-African_hunting_dog/n02116738_2503.jpg'], dtype='<U48')]],
          dtype=object)



    array([[  1],
           [  1],
           [  1],
           ...,
           [120],
           [120],
           [120]], dtype=uint8)




```
train_files.shape,train_targets.shape
```





    ((12000, 1), (12000, 1))





```
# One-hot encoding train targets

train_targets= np.float32(train_targets)-1
train_targets = keras.utils.to_categorical(train_targets, 120)
train_targets
```





    array([[1., 0., 0., ..., 0., 0., 0.],
           [1., 0., 0., ..., 0., 0., 0.],
           [1., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 1.],
           [0., 0., 0., ..., 0., 0., 1.],
           [0., 0., 0., ..., 0., 0., 1.]], dtype=float32)





```
size = 150

def path_to_tensor(img_path):
    # load image
    img = image.load_img('Images/'+str(img_path), target_size=(size, size))
    # convert to 3D tensor with shape (size, size, 3)
    x = image.img_to_array(img)
    #print(x.shape)
    #imshow(img)
    #plt.show()
    # convert 3D tensor to 4D tensor with shape (1, size, size, 3)
    return np.expand_dims(x, axis=0)

def get_tensors(img_list):
    # Get a tensor per image in the list
    list_of_tensors = [path_to_tensor(img_path[0][0]) for img_path in tqdm(img_list)]
    #list_of_tensors=[]
    #for img_path in img_list:
    #    print(img_path[0][0])
    #    list_of_tensors.append(path_to_tensor((img_path[0][0])))
    #print(np.vstack(list_of_tensors))
    return np.vstack(list_of_tensors)
```




```
train_files[0]
```





    array([array(['n02085620-Chihuahua/n02085620_5927.jpg'], dtype='<U38')],
          dtype=object)





```
# Get train data tensors

train_tensors = np.float32(get_tensors(train_files))/255
print('Done')
```


    100%|██████████| 12000/12000 [00:53<00:00, 225.38it/s]


    Done




```
train_tensors.shape
```





    (12000, 150, 150, 3)





```
from keras.preprocessing.image import ImageDataGenerator

# Augmented data image generator
datagen = ImageDataGenerator(
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    horizontal_flip=True) 

# fit image generator on tensors
datagen.fit(train_tensors)
```




```
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization

model = Sequential()
model.add(BatchNormalization(input_shape=(size, size, 3)))
model.add(Conv2D(filters=16, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(GlobalAveragePooling2D())

model.add(Dense(120, activation='softmax'))

model.summary()
```


    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    batch_normalization_11 (Batc (None, 150, 150, 3)       12        
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, 148, 148, 16)      448       
    _________________________________________________________________
    max_pooling2d_9 (MaxPooling2 (None, 74, 74, 16)        0         
    _________________________________________________________________
    batch_normalization_12 (Batc (None, 74, 74, 16)        64        
    _________________________________________________________________
    conv2d_12 (Conv2D)           (None, 72, 72, 32)        4640      
    _________________________________________________________________
    max_pooling2d_10 (MaxPooling (None, 36, 36, 32)        0         
    _________________________________________________________________
    batch_normalization_13 (Batc (None, 36, 36, 32)        128       
    _________________________________________________________________
    conv2d_13 (Conv2D)           (None, 34, 34, 64)        18496     
    _________________________________________________________________
    max_pooling2d_11 (MaxPooling (None, 17, 17, 64)        0         
    _________________________________________________________________
    batch_normalization_14 (Batc (None, 17, 17, 64)        256       
    _________________________________________________________________
    conv2d_14 (Conv2D)           (None, 15, 15, 128)       73856     
    _________________________________________________________________
    max_pooling2d_12 (MaxPooling (None, 7, 7, 128)         0         
    _________________________________________________________________
    batch_normalization_15 (Batc (None, 7, 7, 128)         512       
    _________________________________________________________________
    conv2d_15 (Conv2D)           (None, 5, 5, 256)         295168    
    _________________________________________________________________
    max_pooling2d_13 (MaxPooling (None, 2, 2, 256)         0         
    _________________________________________________________________
    batch_normalization_16 (Batc (None, 2, 2, 256)         1024      
    _________________________________________________________________
    global_average_pooling2d_1 ( (None, 256)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 120)               30840     
    =================================================================
    Total params: 425,444
    Trainable params: 424,446
    Non-trainable params: 998
    _________________________________________________________________




```
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

```




```
epochs = 20
batch_size = 20

checkpointer = ModelCheckpoint(filepath='weights.bestaugmented.hdf5', 
                               verbose=1, save_best_only=True)

# Fitting with data augmentation
model.fit_generator(datagen.flow(train_tensors, train_targets, batch_size=batch_size),
                    #validation_data=(test_tensors, test_targets), 
                    steps_per_epoch=train_tensors.shape[0] // batch_size,
                    epochs=epochs, callbacks=[checkpointer], verbose=1)
```


    Epoch 1/20
    600/600 [==============================] - 78s 130ms/step - loss: 4.5236 - acc: 0.0398
    Epoch 2/20
      1/600 [..............................] - ETA: 1:09 - loss: 4.8717 - acc: 0.0000e+00
    
    /usr/local/lib/python3.6/dist-packages/keras/callbacks.py:434: RuntimeWarning: Can save best model only with val_loss available, skipping.
      'skipping.' % (self.monitor), RuntimeWarning)


    600/600 [==============================] - 74s 123ms/step - loss: 4.1492 - acc: 0.0760
    Epoch 3/20
    600/600 [==============================] - 73s 122ms/step - loss: 3.8975 - acc: 0.1120
    Epoch 4/20
    600/600 [==============================] - 73s 121ms/step - loss: 3.6710 - acc: 0.1423
    Epoch 5/20
    600/600 [==============================] - 72s 121ms/step - loss: 3.4841 - acc: 0.1789
    Epoch 6/20
    600/600 [==============================] - 73s 121ms/step - loss: 3.2815 - acc: 0.2182
    Epoch 7/20
    600/600 [==============================] - 73s 122ms/step - loss: 3.1244 - acc: 0.2463
    Epoch 8/20
    600/600 [==============================] - 73s 121ms/step - loss: 2.9797 - acc: 0.2674
    Epoch 9/20
    600/600 [==============================] - 73s 121ms/step - loss: 2.8404 - acc: 0.3048
    Epoch 10/20
    600/600 [==============================] - 73s 121ms/step - loss: 2.7352 - acc: 0.3156
    Epoch 11/20
    600/600 [==============================] - 73s 122ms/step - loss: 2.6197 - acc: 0.3443
    Epoch 12/20
    600/600 [==============================] - 74s 123ms/step - loss: 2.5078 - acc: 0.3701
    Epoch 13/20
    600/600 [==============================] - 75s 124ms/step - loss: 2.4067 - acc: 0.3842
    Epoch 14/20
    600/600 [==============================] - 74s 124ms/step - loss: 2.3619 - acc: 0.3930
    Epoch 15/20
    600/600 [==============================] - 74s 123ms/step - loss: 2.2596 - acc: 0.4133
    Epoch 16/20
    600/600 [==============================] - 73s 121ms/step - loss: 2.1964 - acc: 0.4380
    Epoch 17/20
    600/600 [==============================] - 72s 120ms/step - loss: 2.1352 - acc: 0.4416
    Epoch 18/20
    600/600 [==============================] - 72s 120ms/step - loss: 2.0783 - acc: 0.4534
    Epoch 19/20
    600/600 [==============================] - 73s 121ms/step - loss: 2.0106 - acc: 0.4708
    Epoch 20/20
    600/600 [==============================] - 73s 121ms/step - loss: 1.9582 - acc: 0.4837





    <keras.callbacks.History at 0x7fa8b054e908>





```
epochs = 20
batch_size = 64

checkpointer = ModelCheckpoint(filepath='weights.bestaugmented.hdf5', 
                               verbose=1)

# Fitting with data augmentation
model.fit_generator(datagen.flow(train_tensors, train_targets, batch_size=batch_size),
                    steps_per_epoch=train_tensors.shape[0] // batch_size,
                    epochs=epochs, callbacks=[checkpointer], verbose=1)
```


    Epoch 1/20
    187/187 [==============================] - 67s 356ms/step - loss: 1.5859 - acc: 0.5790
    
    Epoch 00001: saving model to weights.bestaugmented.hdf5
    Epoch 2/20
    187/187 [==============================] - 66s 352ms/step - loss: 1.4697 - acc: 0.6116
    
    Epoch 00002: saving model to weights.bestaugmented.hdf5
    Epoch 3/20
    187/187 [==============================] - 67s 358ms/step - loss: 1.4176 - acc: 0.6300
    
    Epoch 00003: saving model to weights.bestaugmented.hdf5
    Epoch 4/20
    187/187 [==============================] - 67s 359ms/step - loss: 1.3697 - acc: 0.6354
    
    Epoch 00004: saving model to weights.bestaugmented.hdf5
    Epoch 5/20
    187/187 [==============================] - 68s 361ms/step - loss: 1.3427 - acc: 0.6378
    
    Epoch 00005: saving model to weights.bestaugmented.hdf5
    Epoch 6/20
    187/187 [==============================] - 67s 359ms/step - loss: 1.3118 - acc: 0.6484
    
    Epoch 00006: saving model to weights.bestaugmented.hdf5
    Epoch 7/20
    187/187 [==============================] - 67s 359ms/step - loss: 1.2879 - acc: 0.6488
    
    Epoch 00007: saving model to weights.bestaugmented.hdf5
    Epoch 8/20
    187/187 [==============================] - 67s 358ms/step - loss: 1.2384 - acc: 0.6663
    
    Epoch 00008: saving model to weights.bestaugmented.hdf5
    Epoch 9/20
    187/187 [==============================] - 67s 358ms/step - loss: 1.2331 - acc: 0.6628
    
    Epoch 00009: saving model to weights.bestaugmented.hdf5
    Epoch 10/20
    187/187 [==============================] - 66s 354ms/step - loss: 1.2011 - acc: 0.6647
    
    Epoch 00010: saving model to weights.bestaugmented.hdf5
    Epoch 11/20
    187/187 [==============================] - 66s 356ms/step - loss: 1.1539 - acc: 0.6865
    
    Epoch 00011: saving model to weights.bestaugmented.hdf5
    Epoch 12/20
    187/187 [==============================] - 66s 354ms/step - loss: 1.1464 - acc: 0.6806
    
    Epoch 00012: saving model to weights.bestaugmented.hdf5
    Epoch 13/20
    187/187 [==============================] - 67s 356ms/step - loss: 1.1189 - acc: 0.6880
    
    Epoch 00013: saving model to weights.bestaugmented.hdf5
    Epoch 14/20
    187/187 [==============================] - 66s 354ms/step - loss: 1.1067 - acc: 0.6928
    
    Epoch 00014: saving model to weights.bestaugmented.hdf5
    Epoch 15/20
    187/187 [==============================] - 66s 354ms/step - loss: 1.0683 - acc: 0.6984
    
    Epoch 00015: saving model to weights.bestaugmented.hdf5
    Epoch 16/20
    187/187 [==============================] - 66s 355ms/step - loss: 1.0506 - acc: 0.7025
    
    Epoch 00016: saving model to weights.bestaugmented.hdf5
    Epoch 17/20
    187/187 [==============================] - 66s 354ms/step - loss: 1.0183 - acc: 0.7115
    
    Epoch 00017: saving model to weights.bestaugmented.hdf5
    Epoch 18/20
    187/187 [==============================] - 67s 356ms/step - loss: 1.0024 - acc: 0.7189
    
    Epoch 00018: saving model to weights.bestaugmented.hdf5
    Epoch 19/20
    187/187 [==============================] - 67s 356ms/step - loss: 0.9709 - acc: 0.7253
    
    Epoch 00019: saving model to weights.bestaugmented.hdf5
    Epoch 20/20
    187/187 [==============================] - 67s 357ms/step - loss: 0.9515 - acc: 0.7310
    
    Epoch 00020: saving model to weights.bestaugmented.hdf5





    <keras.callbacks.History at 0x7fa8b054e860>





```
model.load_weights('weights.bestaugmented.hdf5')
```




```
train_files=[]

# Load the test_list.mat to get the list of test files

test_files = io.loadmat('test_list.mat')['file_list']
test_targets = io.loadmat('test_list.mat')['labels']
display(test_files)
display(test_targets)
```



    array([[array(['n02085620-Chihuahua/n02085620_2650.jpg'], dtype='<U38')],
           [array(['n02085620-Chihuahua/n02085620_4919.jpg'], dtype='<U38')],
           [array(['n02085620-Chihuahua/n02085620_1765.jpg'], dtype='<U38')],
           ...,
           [array(['n02116738-African_hunting_dog/n02116738_3635.jpg'], dtype='<U48')],
           [array(['n02116738-African_hunting_dog/n02116738_2988.jpg'], dtype='<U48')],
           [array(['n02116738-African_hunting_dog/n02116738_6330.jpg'], dtype='<U48')]],
          dtype=object)



    array([[  1],
           [  1],
           [  1],
           ...,
           [120],
           [120],
           [120]], dtype=uint8)




```
test_files.shape,test_targets.shape
```





    ((8580, 1), (8580, 1))





```
# One-hot encoding test targets

test_targets= np.float32(test_targets)-1
test_targets = keras.utils.to_categorical(test_targets, 120)
test_targets
```





    array([[1., 0., 0., ..., 0., 0., 0.],
           [1., 0., 0., ..., 0., 0., 0.],
           [1., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 1.],
           [0., 0., 0., ..., 0., 0., 1.],
           [0., 0., 0., ..., 0., 0., 1.]], dtype=float32)





```
# Get test data tensors

test_tensors = np.float32(get_tensors(test_files))/255
```


    100%|██████████| 8580/8580 [00:52<00:00, 161.94it/s]




```
# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```


    Test accuracy: 30.1049%

