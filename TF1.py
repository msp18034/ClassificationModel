import tensorflow as tf
import json
import base64
from io import BytesIO
from PIL import Image
import time
from timeit import default_timer as timer
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def toTFExample(image,label,ingredients):
   # print("toExample")
    example = tf.train.Example(
        features=tf.train.Features(
          feature={
              'image': _bytes_feature(image.tostring()),
            'label': _bytes_feature(label.tostring()),
            'ingres': _bytes_feature(ingredients.tostring())
          } 
        )
      )
    return example.SerializeToString()
def parse_line(line):
    '''
    Given a line from the training/test txt file, return parsed info.'''
    path,ingre=line.split(" ",1)
    classname=path.split("/")[1]
    ingre=np.fromstring(ingre, dtype=int, sep=' ')
    return path, classname, ingre


def main():
    count=0
    writer = tf.python_io.TFRecordWriter("shuffled.tfrecord")
    with tf.Session() as sess:
        with open("shuffled.txt", 'r') as infile:
            for line in infile:
                count+=1
                path, label, ingres=parse_line(line)
                label=np.fromstring(label, dtype=int, sep=' ')
                image=np.array([cv2.imread("/home/hduser/Vireo"+path)])
                
                if label[0]==172:
                    label[0]=0
                example=toTFExample(image,label,ingres)
                writer.write(example)
                if(count%100==0):
                    print(count,"images sended")
                    time.sleep(10)
                if count%500==0:
                    break

            writer.close()

if __name__ == '__main__':
    main()
