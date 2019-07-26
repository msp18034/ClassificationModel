import os
import re
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession 
from bigdl.util.common import *
from pyspark.sql import SQLContext

from bigdl.nn.criterion import *
from bigdl.nn.layer import *
from bigdl.optim.optimizer import Adam

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, udf, substring
from pyspark.sql.types import DoubleType, StringType

from zoo.feature.text import TextSet
from zoo.common.nncontext import *
from zoo.feature.image import *
from zoo.pipeline.api.keras.layers import Dense, Input, Flatten
#from zoo.pipeline.api.keras.models import *
#from zoo.pipeline.api.net import *
from zoo.pipeline.nnframes import *
from zoo.models.common.zoo_model import *


from pyspark.sql import Row

from zoo import init_spark_on_yarn

sc = init_spark_on_yarn(
    conda_name="base", # The name of the created conda-env
    num_executor=2,
    executor_cores=4,
    executor_memory="8g",
    driver_memory="2g",
    driver_cores=2,
    extra_executor_memory_for_ray="2g")
image_path = "hdfs:///Vireo/*"
image_DF = NNImageReader.readImages(image_path, sc).withColumn("id",substring("image.origin",50,9))
print(image_DF.count())
