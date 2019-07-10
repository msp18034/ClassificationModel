import glob
import math
from PIL import Image
import numpy as np
import os
import keras
def read_img(path, target_size):
    try:
        img = Image.open(path).convert("RGB")
        img_rs = img.resize(target_size)
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
    classname=path.split("/")[1]
    ingre=np.fromstring(ingre, dtype=int, sep=' ')
    return "/home/student/VireoFood172"+path

def parse_label(line):
    '''
    Given a line from the training/test txt file, return parsed info.'''
    path,ingre=line.split(" ",1)
    classname=path.split("/")[1]
    ingre=np.fromstring(ingre, dtype=int, sep=' ')
    return classname


def my_gen(path, batch_size, target_size):
    img_list =getList("full.txt") 
    steps = math.ceil(len(img_list) / batch_size)
    print("Found %s images."%len(img_list))
    while True:
        for i in range(steps):
            batch_list = img_list[i * batch_size : i * batch_size + batch_size]
            x=[parse_path(line) for line in batch_list] 
            x = [read_img(file, target_size) for file in x]
            batch_x = np.concatenate([array for array in x])
            
            y = [ parse_label(line) for line in batch_list]
            y= keras.utils.to_categorical(y,1000)
            yield batch_x, y   

from keras.applications import ResNet50
from keras import optimizers

path = 'VireoFood172/1/'
model = ResNet50()
model.compile(optimizer=optimizers.Adam(1e-4), loss='categorical_crossentropy')

batch_size = 64
steps = 94
target_size = (224, 224)
data_gen = my_gen(path, batch_size, target_size)  


model.fit_generator(data_gen, steps_per_epoch=steps, epochs=10, verbose=1, 
                    use_multiprocessing=True, workers=2)
