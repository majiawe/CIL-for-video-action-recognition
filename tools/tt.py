import os
import numpy as np
from multiprocessing import Pool
from heapq import nsmallest
import pickle

res = {}

def split_seq(line):
    global res
    line = line.split(' ')


    if len(line) == 1:
        return -1

    elif len(line) == 4 and len(line[0].split('\t')) == 3:
        line1 = [*line[0].split('\t'), line[1].split(',')[0], line[2].split(',')[0], line[3].split('\n')[0]]
        line1.pop(1)
    elif len(line) == 8 and line[1] == '' and len(line[4].split('\t')) == 3:
        line.pop(1)
        line.pop(1)
        line.pop(1)
        line = [line[0] + '\t' + line[1],line[2],line[3],line[4]]
        line1 = [*line[0].split('\t'), line[1].split(',')[0], line[2].split(',')[0], line[3].split('\n')[0]]
        line1.pop(1)
        line1.pop(1)
    elif len(line) == 8 and line[1] == '' and len(line[4].split('\t')) == 4:
        line.pop(1)
        line.pop(1)
        line.pop(1)
        line = [line[0] + line[1], line[2], line[3],line[4]]
        line1 = [*line[0].split('\t'), line[1].split(',')[0], line[2].split(',')[0], line[3].split('\n')[0]]
        line1.pop(1)
        line1.pop(1)
    elif len(line) == 7 and line[1] == '' and len(line[4].split('\t')) == 4:
        line.pop(1)
        line.pop(1)
        line.pop(1)
        line = [line[0]  + line[1], line[2], line[3]]
        line1 = [*line[0].split('\t'), line[1].split(',')[0], line[2].split('\n')[0]]
        line1.pop(1)
        line1.pop(1)
    elif len(line) == 3 and len(line[0].split('\t')) == 4:
        line = [*line[0].split('\t'), line[1].split(',')[0] ,line[2].split('\n')[0]]
        line.pop(1)
        line.pop(1)
        line1 = line
    else:
        line1 = [*line[0].split('\t'),line[1].split(',')[0],line[2].split(',')[0],line[3].split('\n')[0]]
        line1.pop(1)
        line1.pop(1)
    # elif line[0] == 'person11_boxing_d4' or (line[1] == '' and len(line) == 7) :
    #     line.pop(1)
    #     line.pop(1)
    #     line.pop(1)
    #     line = [line[0]  + line[1], line[2], line[3]]
    #     line1 = [*line[0].split('\t'), line[1].split(',')[0], line[2].split('\n')[0]]
    #     line1.pop(1)
    #     line1.pop(1)
    #
    # elif line[0] == 'person11_jogging_d4' or (line[1] == '' and len(line) == 8) or line[0] == 'person14_jogging_d4' or line[0] == 'person11_running_d4'or line[0] == 'person11_walking_d4'or line[0] == 'person12_boxing_d4' or line[0] == 'person14_boxing_d4':
    #     line.pop(1)
    #     line.pop(1)
    #     line.pop(1)
    #     line = [line[0] + line[1], line[2], line[3],line[4]]
    #     line1 = [*line[0].split('\t'), line[1].split(',')[0],line[2].split(',')[0] ,line[3].split('\n')[0]]
    #     line1.pop(1)
    #     line1.pop(1)
    #     line1.pop(1)
    # elif line[0] == 'person13_handwaving_d3\t\tframes\t1-125,' or len(line) == 4 or line[0] == 'person14_handwaving_d3\t\tframes\t1-145,':
    #     line = [*line[0].split('\t'),line[1].split(',')[0] ,line[2].split(',')[0],line[3].split('\n')[0]]
    #     line.pop(1)
    #     line.pop(1)
    #     line1 = line
    #
    #
    # elif line[1] == '' :
    #     line.pop(1)
    #     line.pop(1)
    #     line.pop(1)
    #     line = [line[0] + '\t' + line[1],line[2],line[3],line[4]]
    #     line1 = [*line[0].split('\t'), line[1].split(',')[0], line[2].split(',')[0], line[3].split('\n')[0]]
    #     line1.pop(1)
    #     line1.pop(1)


    print(line1)
    for i,d in enumerate(line1):
        if i == 0:
            res[d] = []
            name = d
        elif i == 1:
            num = d.split(',')[0]
            num = num.split('-')
            num = [int(l) for l in num]
            res[name].append(num)
        else:
            num = d.split('-')
            num = [int(l) for l in num]
            res[name].append(num)

    return line1

with open('/data1/ma_gps/KTH_dataset/sequences','r') as f:
    for line in f.readlines():
        if line == '\n' :
            pass
        else:
            split_seq(line)
    print(res)
    # data = f.read()
    # data = data.split('\n')
    # print(data)

with open('../split_video.pkl', 'wb') as f:
    pickle.dump(res,f)