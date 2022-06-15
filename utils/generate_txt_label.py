import os,sys,json
import torch
import numpy as np
import pandas as pd
from PIL import Image
def generate_train(path):
    json_file=os.path.join(path,"../../dg_label_id_mapping.json")
    with open(json_file,"r") as f:
        result=[]
        class_dict=json.load(f)
        for (root,dirs,files) in os.walk(path):
            for i,dir in enumerate(dirs):
                root_dir=os.path.join(root,dir)
                for (droot,ddirs,dfiles) in os.walk(root_dir):
                    for ddir in ddirs:
                        class_name=ddir
                        class_label=class_dict[class_name]
                        root_ddir=os.path.join(droot,ddir)
                        for (ddroot,dddirs,ddfiles) in os.walk(root_ddir):
                            for ddfile in ddfiles:
                                img_path=os.path.join(ddroot,ddfile)
                                result.append([img_path,class_label,i]) # img_path, class_label, domain
        print(len(result))
        return result

def generate_test(path):
    result=[]
    for (root,dirs,files) in os.walk(path):
        for file in files:
            img_path=os.path.join(root,file)
            result.append(img_path)
    return result




