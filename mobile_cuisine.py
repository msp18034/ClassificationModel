from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.layers import Input
from keras.optimizers import SGD
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model,load_model
from keras.layers import Dense, GlobalAveragePooling2D,Dropout,Flatten,BatchNormalization
from keras import backend as K
import keras
import numpy as np
import argparse
import math
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import pickle
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--epoch', type=int,
                    help='set input ecpoch')
parser.add_argument('--reload', type=int,
                    help='reload or train from 0')
parser.add_argument('--train', type=int,
                    help='train 1,evaluate 0 or whatever')
parser.add_argument('--model', type=str,
                    help='model path')


args = parser.parse_args()

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

def read_val_img(path, target_size):
    try:
        img = Image.open(path).convert("RGB")
        img_rs = img.resize(target_size)
#       img_rs = img_rs*1.0/255
    except Exception as e:
        print(e)
    else:
        x =np.array(img_rs)
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
    doc={0: 20, 1: 1, 2: 0, 3: 3, 4: 4, 5: 4, 6: 0, 7: 0, 8: 0, 9: 1, 10: 3, 11: 0, 12: 0, 13: 1, 14: 6, 15: 4, 16: 8, 17: 9, 18: 0, 19: 0, 20: 10, 21: 9, 22: 3, 23: 0, 24: 6, 25: 5, 26: 10, 27: 8, 28: 5, 29: 9, 30: 6, 31: 6, 32: 1, 33: 9, 34: 10, 35: 8, 36: 3, 37: 8, 38: 4, 39: 4, 40: 1, 41: 4, 42: 12, 43: 7, 44: 0, 45: 0, 46: 0, 47: 0, 48: 6, 49: 0, 50: 3, 51: 0, 52: 9, 53: 11, 54: 9, 55: 1, 56: 5, 57: 5, 58: 8, 59: 8, 60: 12, 61: 12, 62: 0, 63: 1, 64: 1, 65: 6, 66: 0, 67: 0, 68: 0, 69: 4, 70: 6, 71: 0, 72: 0, 73: 12, 74: 0, 75: 0, 76: 6, 77: 6, 78: 0, 79: 6, 80: 11, 81: 0, 82: 6, 83: 11, 84: 9, 85: 9, 86: 4, 87: 0, 88: 12, 89: 8, 90: 8, 91: 7, 92: 1, 93: 0, 94: 0, 95: 5, 96: 0, 97: 0, 98: 0, 99: 0, 100: 12, 101: 8, 102: 6, 103: 2, 104: 2, 105: 0, 106: 1, 107: 1, 108: 1, 109: 11, 110: 4, 111: 8, 112: 11, 113: 8, 114: 8, 115: 4, 116: 7, 117: 9, 118: 5, 119: 5, 120: 5, 121: 5, 122: 8, 123: 1, 124: 10, 125: 10, 126: 10, 127: 0, 128: 12, 129: 9, 130: 12, 131: 7, 132: 9, 133: 4, 134: 8, 135: 8, 136: 6, 137: 8, 138: 8, 139: 12, 140: 12, 141: 10, 142: 8, 143: 8, 144: 8, 145: 4, 146: 4, 147: 0, 148: 0, 149: 12, 150: 6, 151: 6, 152: 11, 153: 4, 154: 8, 155: 4, 156: 0, 157: 4, 158: 4, 159: 4, 160: 0, 161: 6, 162: 11, 163: 0, 164: 5, 165: 10, 166: 8, 167: 4, 168: 6, 169: 6, 170: 0, 171: 5, 172: 4}
    return np.array(classname),np.array(doc[int(classname)])

def parse_ingres(line):
    '''
    Given a line from the training/test txt file, return parsed info.'''
    path,ingre=line.split(" ",1)
    ingre=np.fromstring(ingre, dtype=int, sep=' ')
    return np.expand_dims(ingre, axis=0)

def create_model(ing_num,classes):
    # create the base pre-trained model
    #base_model = InceptionV3(weights='imagenet', include_top=False)
    Inp = Input((224, 224, 3))
    #base_model = ResNet50(weights='imagenet', include_top=False,                         input_shape=(256, 256, 3), )
    base_model=keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False )

    K.set_learning_phase(1)
    x = base_model(Inp)
    x = BatchNormalization(axis=-1, name='banNew')(x)
    x = GlobalAveragePooling2D(name='average_pool')(x)
    #x = Flatten(name='flatten')(x)
    #x= Dense(4096, activation='relu', name="fc1")(x)
    #x = Dropout(0.5)(x)
    cuisine = Dense(1024, activation='relu', name="fc2")(x)
    cuisine = Dropout(0.5)(cuisine)
    cuisine = Dense(13, activation='softmax', name="cuisine")(cuisine)

    #merged_vector = keras.layers.concatenate([x, ingredients], axis=-1)
    predictions = Dense(4096, activation='relu', name="fc3")(x)
    predictions = Dropout(0.5)(predictions)
    predictions = Dense(classes, activation='softmax', name="predictions")(predictions)

    input_tensor = Input(shape=(400, 400, 3))  # this assumes K.image_data_format() == 'channels_last'
    model = Model(input=Inp, output=[cuisine, predictions])       
 #  model = Model(input=base_model.input, output=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    return model


def my_gen(path,nbclass, batch_size, target_size,mode):
    img_list =getList(path)
    #steps=500
    steps = math.ceil(len(img_list) / batch_size)
    print("Found %s images."%len(img_list))
    while True:
        for i in range(steps):
            batch_list = img_list[i * batch_size : i * batch_size + batch_size]
    
            x=[parse_path(line) for line in batch_list]
            x = [read_img(file, target_size) for file in x]
            batch_x = np.concatenate([array for array in x])
            y = [ parse_label(line)[0] for line in batch_list]
            batch_y= keras.utils.to_categorical(y,nbclass)
            cuisine=[parse_label(line)[1] for line in batch_list]
            batch_c=keras.utils.to_categorical(cuisine,13)

            #ingres = [ parse_ingres(line) for line in batch_list]
            #batch_ingres= np.concatenate([array for array in ingres])  
            if(mode=="train"):
                a = image_gen.flow(batch_x,batch_y,batch_size = batch_x.shape[0],shuffle=False)
                x,y=next(a)
                yield x,[batch_c,y]
            else:
              #  a = valid_gen.flow(batch_x,batch_y,batch_size = batch_x.shape[0],shuffle=False)
               # x,y=next(a)
                #print("image after ---------------",x[0])
                batch_x=batch_x*1.0/255
                yield batch_x,[batch_c,batch_y]

def read_val():
    f = open("val.txt",'r')
    images=[]
    ingres=[]
    classes=[]
    count=0
    for line in f:
        path,ingre=line.split(" ",1)
        classname=path.split("/")[1]
        ingre=np.fromstring(ingre, dtype=int, sep=' ')
        #print("/home/student/VireoFood172"+path)
        try:
            image=read_val_img("VireoFood172"+path,(256,256))
            image=image*1.0/255
            count+=1
            images.append(image)
            ingres.append(ingre)
            classes.append(classname)
            if count%500==0:
                print(count,"images readed")
        except Exception as e:
            pass
        if count%1000==0:
            break
    images=np.array(images)
    classes=np.array(classes)
    y_train = keras.utils.to_categorical(classes,173)
    ingres=np.array(ingres)
    print(y_train.shape)
    print(ingres.shape)
    return images,[ingres,y_train]

train_path="train.txt"
val_path="val.txt"
batch_size=64
nbclass=173
steps=math.ceil(len(getList(train_path)) / batch_size/2)
val_steps=math.ceil(len(getList(val_path)) / batch_size/2)
target_size = (224,224)

image_gen=ImageDataGenerator(rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

valid_gen=ImageDataGenerator(rescale=1./255)

train_gen =my_gen(train_path,173, batch_size, target_size,"train")

val_gen=my_gen(val_path,173, batch_size, target_size,"val")

model_path=args.model

if args.reload==0:
    #odel_path="model.h5"
    model=create_model(353,nbclass)
    model.compile(
            loss={
                'cuisine': 'categorical_crossentropy',
                'predictions': 'categorical_crossentropy'
                  },
            loss_weights={
                'cuisine': 0.1,
                'predictions': 0.9
                },
            #optimizer='adam',
            optimizer=SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True),

            metrics=['accuracy','top_k_categorical_accuracy'])
else:
    model=keras.models.load_model(model_path)

if args.train==1: 
    history=model.fit_generator(generator=train_gen, steps_per_epoch=200, epochs=args.epoch,validation_data=val_gen,validation_steps=100, verbose=1,use_multiprocessing=True, workers=1)
    model.save(model_path)

    with open(args.model+str(args.epoch), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

else:
    model.evaluate_generator(val_gen,steps=50,verbose=1)


