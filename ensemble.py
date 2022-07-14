import json
from data import write_result
import numpy as np
import torch,os,sys
import torch.nn as nn


def load_json(path):
    with open(path) as file:
        dict = json.load(file)
    return dict

def find_most_frequent(x:list):
    return max(set(x),key=x.count)

def load_prediction_and_vote(predictions_path="./prediction/"):
    predictions = [os.path.join(predictions_path,x) for x in os.listdir(predictions_path) if x.endswith('json')]
    result=[]
    for i in predictions:
        result.append(load_json(i))
    final={}
    keys=result[0].keys()
    for k in list(keys):
        r = []
        for p in result:
            r.append(p[k])
        r = find_most_frequent(r)
        final[k] = r
    write_result(final,"final_prediction.json")


if __name__=="__main__":
    load_prediction_and_vote()
