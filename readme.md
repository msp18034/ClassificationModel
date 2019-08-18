## Dataset VireoFood172
Download [Here](http://vireo.cs.cityu.edu.hk/VireoFood172/)
Folder structure
-VireoFood172<br>
-ClassificationModel<br>

## Multi-task DCNN using cuisne method classification as auxiliary task
Train
```
python ClassificationModel/multi_cuisine.py --reload 0 --train 1 --baseModel resnet50 --epoch 1 --model res50_cuisine.h5
```
Test
```
python ClassificationModel/multi_cuisine.py --reload 1 --train 0  --model res50_cuisine.h5
```
## Multi-task DCNN using ingredients reccgnition as auxiliary task
Train
```
python ClassificationModel/multi_ingredients.py --reload 0 --train 1 --baseModel resnet50 --epoch 1 --model res50_ingredients.h5
```
Test
```
python ClassificationModel/multi_ingredients.py --reload 1 --train 0  --model res50_ingredients.h5
```
## Single Task DCNN using resnet50 as base Model
Train
```
 python ClassificationModel/singleRes50.py --reload 0 --train 1 --baseModel --epoch 100 --model singleRes50.h5
```
Test
```
python ClassificationModel/singleRes50.py --reload 1 --train 0 --model singleRes50.h5
```
## Help:
```
--epoch          number of epoches to train the model
--baseModel      mobilenet,inceptionV3 or resnet50. base model as feature extractor
--reload         0,1. 0: train from scratch; 1: train based on previous trained model or load model for evaluation
--train          0,1. 0: evaluate;1: train
--model          string,model path to save trained model or to reload for evaluation
```
