# facereognition

Step 1--> First we make datadets for our model (by the use of HaarCascade).We store our dataset in a file called face

This file is further divided into train and test sets

Train has two subfiles n1 and n2 .n1 has the face data of me and n2 has the face data of my friend.

Simialarly we do it for Test 

Step 2--> we build model and train it using Vgg16 by applying concepts of Transfer Learning and Fine Tuning

here is a walk through of the code

Import the required modules

from keras.applications import vgg16
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.optimizers import RMSprop
from keras.models import load_model
from PIL import Image
import numpy as np

# Then we download weights and build our initial model

model = vgg16.VGG16(weights='imagenet',include_top = False)

# we go on with fine tuning

leaving out the first layer we make out layer trainable which trainable previously

for l in model.layers:
 l.trainable = False
 
# Now we add additional fully connected heads to our model

top_model = model.output
top_model = GlobalAveragePooling2D()(top_model)
top_model = Dense(1024,activation='relu')(top_model)
top_model = Dense(1024,activation='relu')(top_model)
top_model = Dense(2,activation='softmax')(top_model)

# Now our previous is inflated with our newly added layers and the resulting model is as follows:

nmodel = Model(inputs = model.input, outputs = top_model)

# Get train and test Data
img_rows, img_cols = 224,224
from keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'faceData/train/'
validation_data_dir = 'faceData/test/'

# Let's use some data augmentaiton 

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
# set our batch size 
batch_size = 32
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        class_mode='categorical')
Train the model
from keras.optimizers import RMSprop

nmodel.compile(loss = 'categorical_crossentropy'
              ,optimizer = RMSprop(lr = 0.001), metrics = ['accuracy'])


#Enter the number of training and validation samples here

# We only train 5 EPOCHS 
epochs = 5

history = nmodel.fit_generator(
    train_generator,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size)

# Test Model  :

from keras.models import load_model
from PIL import Image
import numpy as np
classifier = load_model('faceRecog.h5')
input_im = Image.open("faceData/test_tushar.jpg")
input_im.show()
input_original = input_im.copy()

input_im = input_im.resize((224, 224))
display(input_im)
input_im = np.array(input_im)
input_im = input_im / 255.
input_im = input_im.reshape(1,224,224,3)
res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)

if res == [0]:
    print('n1 : NAME')
elif res == 1:
    print('n2 : NAME')
