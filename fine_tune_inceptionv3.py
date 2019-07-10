import cv2
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras.optimizers import SGD
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model,load_model
from keras.layers import Dense, GlobalAveragePooling2D,Dropout
from keras import backend as K
import keras
import numpy as np
import argparse
import math
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
#parser = argparse.ArgumentParser(description='Process some integers.')
#parser.add_argument('count', type=int,
                    #help='an integer for the accumulator')

#args = parser.parse_args()

def read_img(path, target_size):
    try:
        img = Image.open(path).convert("RGB")
        img_rs = img.resize(target_size)
#        img_rs = img_rs*1.0/255
    except Exception as e:
        print(e)
    else:
        x = np.expand_dims(np.array(img_rs), axis=0)
        return x

def getList(path):
    lines=[]
    with open(path, 'r') as infile:
        for line in infile:
            if line is not None:
                lines.append(line)
    return lines

def parse_path(line):
    '''
    Given a line from the training/test txt file, return parsed info.'''
    path,ingre=line.split(" ",1)
    return "VireoFood172"+path

def parse_label(line):
    '''
    Given a line from the training/test txt file, return parsed info.'''
    path,ingre=line.split(" ",1)
    classname=path.split("/")[1]
    return np.array(classname)

def parse_ingres(line):
    '''
    Given a line from the training/test txt file, return parsed info.'''
    path,ingre=line.split(" ",1)
    ingre=np.fromstring(ingre, dtype=int, sep=' ')
    return np.expand_dims(ingre, axis=0)

def create_model(ing_num,classes):
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu', name="fc1")(x)
    x = Dropout(0.5)(x)
    ingredients = Dense(2048, activation='relu', name="fc2")(x)
    ingredients = Dropout(0.5)(ingredients)
    ingredients = Dense(ing_num, activation='sigmoid', name="ingredients")(ingredients)

    #merged_vector = keras.layers.concatenate([x, ingredients], axis=-1)
    predictions = Dense(2048, activation='relu', name="fc3")(x)
    predictions = Dropout(0.5)(predictions)
    predictions = Dense(classes, activation='softmax', name="predictions")(predictions)

    input_tensor = Input(shape=(400, 400, 3))  # this assumes K.image_data_format() == 'channels_last'
    model = Model(input=base_model.input, output=[ingredients, predictions])       
 #  model = Model(input=base_model.input, output=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    return model


def my_gen(path,nbclass, batch_size, target_size):
    img_list =getList(path)
    steps=1
   # steps = math.ceil(len(img_list) / batch_size)
    print("Found %s images."%len(img_list))
    while True:
        for i in range(steps):
            batch_list = img_list[i * batch_size : i * batch_size + batch_size]
    
            x=[parse_path(line) for line in batch_list]
            x = [read_img(file, target_size) for file in x]
            batch_x = np.concatenate([array for array in x])
            
            y = [ parse_label(line) for line in batch_list]
            batch_y= keras.utils.to_categorical(y,nbclass)
            
            ingres = [ parse_ingres(line) for line in batch_list]
            batch_ingres= np.concatenate([array for array in ingres])            
            yield batch_x,batch_ingres,batch_y

image_gen=ImageDataGenerator(rescale=1./255,featurewise_center=True,rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

def create_aug_gen(in_gen):
    for in_x,batch_ingres,in_y in in_gen:
        #print("image before---------------",in_x[0])
        a = image_gen.flow(in_x,in_y,batch_size = in_x.shape[0],shuffle=False) 
        x,y=next(a)
        #print("image after ---------------",x[0])
        yield x,[batch_ingres,y]


def create_val_df(filePath):
    paths=[]
    labels=[]
    with open(filePath, 'r') as infile:
        for line in infile:
            path,ingre=line.split(" ",1)
            label=path.split("/")[1]
            paths.append(path)
            labels.append(label)
    diction={'paths':paths,'labels':labels}
    return pd.DataFrame(diction)
train_path="train.txt"
val_path="val.txt"
batch_size=12
nbclass=173
steps=math.ceil(len(getList(train_path)) / batch_size)
val_steps=math.ceil(len(getList(val_path)) / batch_size)
target_size = (256,256)

image_gen=ImageDataGenerator(rescale=1./255,featurewise_center=True,rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
train_gen =create_aug_gen(my_gen(train_path,173, batch_size, target_size))

valid_df=create_val_df("val.txt")

valid_gen=ImageDataGenerator(rescale=1./255,featurewise_center=True)
val_gen=valid_gen.flow_from_dataframe(dataframe=valid_df,dictionary="/home/student/VireoFood172",x_col="paths",y_col="labels",class_mode="categorical", target_size=(256,256),batch_size=32)

model_path="model.h5"
model=create_model(353,nbclass)
model.compile(
            loss={
                'ingredients': 'binary_crossentropy',
                'predictions': 'categorical_crossentropy'
                  },
            loss_weights={
                'ingredients': 1.,
                'predictions': 2.
                },
            #optimizer='adam',
            optimizer=SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
            metrics=['accuracy'])
#model.fit(images,classes, batch_size=32,epochs=10)
#history=model.fit(images,[ingres,y_train], batch_size=32,validation_split=0.1,epochs=100)
model.fit_generator(generator=train_gen, steps_per_epoch=steps, epochs=75, verbose=1,validation_data=val_gen,validation_steps=val_steps,use_multiprocessing=True, workers=1)

model.save(model_path)
