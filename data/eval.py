#!/usr/bin/env python
#coding=utf8
'''
Copyright 2018 Zhihu. ALl Rights Reserved
Author: huangbo@zhihu.com
Creat date: 20180724
description: 
'''
import sys,os,commands,re

import csv 

def hash(k):
    return ((8765 * k + 1234) %  10000000019) % 10000000001

    
true_set = set()

for line in open('./hashed_test_labels'):
    true_set.add(int(line.strip()))

all = len(true_set) * 1.0
acc = 0.0
for items in csv.reader(open(sys.argv[1], 'rb')):
    hashed_res = hash(int(items[0]) + int(items[1]))

    if hashed_res in true_set:
        acc += 1

print acc / all
