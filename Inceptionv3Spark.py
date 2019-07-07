from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.conf import SparkConf

import argparse
import numpy
import tensorflow as tf
from datetime import datetime

from tensorflowonspark import TFCluster
images = sc.newAPIHadoopFile("hdfs:///shuffled.tfrecord", "org.tensorflow.hadoop.io.TFRecordFileInputFormat", keyClass="org.apache.hadoop.io.BytesWritable",valueClass="org.apache.hadoop.io.NullWritable")

def parse_record(bytestr):
    example = tf.train.Example()
    example.ParseFromString(bytestr)
    features = example.features.feature
    label = np.array(features['label'].int64_list.value)
    shape = np.array(features['shape'].int64_list.value)[0]
    image = np.array(features['image'].int64_list.value).reshape(shape)
    ingres = np.array(features['ingredients'].int64_list.value)
    return (image, label,ingres)

dataRDD = images.map(lambda x: parse_record(bytes(x[0])))

