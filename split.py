import os
import random

def split():
    trainList=[]
    valList=[]
    testList=[]
         
    count=0
    with open("full.txt", 'r') as infile:
        for line in infile:
            if line is not None:
                count+=1
                rnd=random.random()
                if rnd<0.8:
                    trainList.append(line)
                elif rnd>=0.8 and rnd<0.95:
                    valList.append(line)
                else:
                    testList.append(line)

    print(len(trainList))
    train = open("train.txt",'w')

    for line in trainList:
        train.write(line)
    val= open("val.txt",'w')
    for line in valList:
        val.write(line)
    test=open("test.txt",'w')
    for line in testList:
        test.write(line)

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
#huffle()
split()
