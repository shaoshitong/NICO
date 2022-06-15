import torch
import torch.nn as nn
import torch.nn.functional as F
import json
def compare_difference(a,b):
    a=open(a, 'r')
    a=json.load(a)
    b=open(b, 'r')
    b=json.load(b)
    keys=a.keys()
    different_result=[]
    for key in keys:
        if a[key]!=b[key]:
            different_result.append(key)
    return different_result

print(compare_difference("../results/67.json","../results/76.json"))
