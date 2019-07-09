import os
import random

def split():
    trainList=[]
    valList=[]
         
    count=0
    with open("full.txt", 'r') as infile:
        for line in infile:
            if line is not None:
                count+=1
                if count<88192:
                    trainList.append(line)
                else:
                    valList.append(line)

    print(len(trainList))
    train = open("train.txt",'w')

    for line in trainList:
        train.write(line)
    val= open("val.txt",'w')
    for line in valList:
        val.write(line)
def shuffle():
    out = open("full.txt",'w')
    lines=[]
    with open("IngreLabel.txt", 'r') as infile:
        for line in infile:
            if line is not None:
                lines.append(line)
    random.shuffle(lines)
    print(len(lines))
    for line in lines:
        out.write(line)
shuffle()
split()
