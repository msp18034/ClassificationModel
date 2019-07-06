from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
import base64
from io import BytesIO
from PIL import Image
import time
from timeit import default_timer as timer
import numpy as np

class Kafka_producer():
    """使用kafka的生产模块"""

    def __init__(self, kafkahost, kafkaport, kafkatopic):
        self.kafkaHost = kafkahost
        self.kafkaPort = kafkaport
        self.kafkatopic = kafkatopic
        self.producer = KafkaProducer(bootstrap_servers = '{kafka_host}:{kafka_port}'.format(
            kafka_host=self.kafkaHost,
            kafka_port=self.kafkaPort
        ))

    def sendjsondata(self, result):
        try:
            parmas_message = json.dumps(result)
            producer = self.producer
            producer.send(self.kafkatopic, parmas_message.encode('utf-8'))
            producer.flush()
        except KafkaError as e:
            print(e)

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
    producer = Kafka_producer("G4master", 9092, "tfrecord")
    count=0
    with open("/home/hduser/shuffled.txt", 'r') as infile:
        for line in infile:
            path, label, ingres=parse_line(line)
            count+=1
            image = image_to_base64("/home/hduser/Vireo"+path)
            result = {
                'label':label,
                'ingres':ingres.tolist(),
                'image': image,
                'count':count
                }
            producer.sendjsondata(result)
            if(count%100==0):
                print(count,"images sended")
                time.sleep(10)


if __name__ == '__main__':
    main()
