## Multi-task DCNN using cuisne method classification as auxiliary task
Train
```
python ClassificationModel/fine_tune_cusine.py --reload 0 --train 1 --baseModel inceptionV3 --epoch 1 --model mobile
```
Test
```
python ClassificationModel/fine_tune_cusine.py --reload 1 --train 0  --model resnetC.h5
```
## Multi-task DCNN using ingredients reccgnition as auxiliary task
Train
```
python ClassificationModel/fine_tune_cusine.py --reload 0 --train 1 --baseModel inceptionV3 --epoch 1 --model mobile
```
Test
```
python ClassificationModel/fine_tune_cusine.py --reload 1 --train 0  --model resnetC.h5
```
## Single Task DCNN using resnet50 as base Model
Train
```
 python ClassificationModel/singleRes50.py --reload 0 --train 1 --baseModel --epoch 100 --model singleRes50.h5
```
Test
```

