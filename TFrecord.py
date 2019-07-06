from kafka import KafkaProducer
from kafka.errors import KafkaError
import tensorflow as tf
import json
import base64
from io import BytesIO
from PIL import Image
import time
from timeit import default_timer as timer
import numpy as np
import cv2

def toTFExample(image,shape,label,ingredients):
   # print("toExample")
    example = tf.train.Example(
        features=tf.train.Features(
          feature={
            'image': tf.train.Feature(int64_list=tf.train.Int64List(value=image.astype("int64"))),
            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=shape)),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label.astype("int64"))),
            'ingredients':tf.train.Feature(int64_list=tf.train.Int64List(value=ingredients.astype("int64")))
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

def image_to_base64(image_path):
    img = Image.open(image_path)
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    #print(base64_str)
    return base64_str

def main():
    count=0
    writer = tf.python_io.TFRecordWriter("shuffled.tfrecord")
    with tf.Session() as sess:
        with open("/home/hduser/shuffled.txt", 'r') as infile:
            for line in infile:
                count+=1
                path, label, ingres=parse_line(line)
                label=np.fromstring(label, dtype=int, sep=' ')
                image=cv2.imread("/home/hduser/Vireo"+path)
                shape=image.shape
                if label[0]==172:
                    label[0]=0

                image=image.flatten()
                example=toTFExample(image, shape,label,ingres)
                writer.write(example)
                if(count%100==0):
                    print(count,"images sended")
                    time.sleep(10)
                #if count%500==0:
                 #   break

            writer.close()

if __name__ == '__main__':
    main()
