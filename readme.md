### Dataset VireoFood172

![avatar](https://github.com/msp18034/ClassificationModel/blob/master/images/sample.png)

Download [Here](http://vireo.cs.cityu.edu.hk/VireoFood172/)<br>



### Label Cooking method Examples

![label](https://github.com/msp18034/ClassificationModel/blob/master/images\cuisine.png)



### Multi-Task DCNN Architecture

![Arch](https://github.com/msp18034/ClassificationModel/blob/master/images\multi-task.png)

### Train and evaluate command

+ #### Multi-task DCNN using cuisine method classification as auxiliary task

Train
```
python ClassificationModel/multi_cuisine.py --reload 0 --train 1 --baseModel resnet50 --epoch 1 --model res50_cuisine.h5
```
Test
```
python ClassificationModel/multi_cuisine.py --reload 1 --train 0  --model res50_cuisine.h5
```
+ #### Multi-task DCNN using ingredients recognition as auxiliary task

Train
```
python ClassificationModel/multi_ingredients.py --reload 0 --train 1 --baseModel resnet50 --epoch 1 --model res50_ingredients.h5
```
Test
```
python ClassificationModel/multi_ingredients.py --reload 1 --train 0  --model res50_ingredients.h5
```
+ #### Single Task DCNN using resnet50 as base Model

Train
```
 python ClassificationModel/singleRes50.py --reload 0 --train 1 --baseModel --epoch 100 --model singleRes50.h5
```
Test
```
python ClassificationModel/singleRes50.py --reload 1 --train 0 --model singleRes50.h5
```
+ ##### Help:

```
--epoch          number of epoches to train the model
--baseModel      mobilenet,inceptionV3 or resnet50. base model as feature extractor
--reload         0,1. 0: train from scratch; 1: train based on previous trained model or load model for evaluation
--train          0,1. 0: evaluate;1: train
--model          string,model path to save trained model or to reload for evaluation
```



### Model Evaluation

![Evaluation](https://github.com/msp18034/ClassificationModel/blob/master/images\eva.png)

