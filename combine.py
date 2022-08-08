import json
from data import write_result
import numpy as np
import torch,os,sys
import torch.nn as nn


if __name__=="__main__":
    path1="kfinal_prediction2.json"
    path2="kkfinal_prediction.json"
    path3="./c.json"
    with open(path1, "r") as f:
        dict1 = json.load(f)
    with open(path2, "r") as f:
        dict2 = json.load(f)
    dict1.update(dict2)
    with open(path3, "w") as f:
        json.dump(dict1,f)
    print("managed to save result!")
    print("-" * 100)