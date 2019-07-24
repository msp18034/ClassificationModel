from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.conf import SparkConf

import argparse
import numpy
import tensorflow as tf
from datetime import datetime
from zoo import init_spark_on_yarn

sc = init_spark_on_yarn(
    hadoop_conf="/opt/hadoop-2.7.5/etc/hadoop",
    conda_name="base",# The name of the created conda-env
    num_executor=2,
    executor_cores=4,
    executor_memory="2g",
    driver_memory="2g",
    driver_cores=2,
    extra_executor_memory_for_ray="2g")
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
    return image, [label,ingres]

dataRDD = images.map(lambda x: parse_record(bytes(x[0])))

