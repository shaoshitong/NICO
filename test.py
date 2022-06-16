
import torch
import torch.nn as nn
from tqdm import tqdm

import model
from data.dataUtils import write_result
import helper


def test(loader, model,epoch=10,cutmix_prob=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    result = {}
    with torch.no_grad():
        for j in tqdm(range(epoch)):
            for i,(x,p,name) in enumerate(loader):
                x = x.to(device)
                y = model(x).softmax(1)
                for j, subname in enumerate(list(name)):
                    subname=subname.split("/")[-1]
                    if subname in result.keys():
                        result[subname]+=(y[j,:].detach()*(p[j]+cutmix_prob))
                    else:
                        result[subname]=(y[j,:].detach()*(p[j]+cutmix_prob))
    for name in result.keys():
        result[name]=result[name].argmax(0).item()
    write_result(result)

if __name__=="__main__":
    args = helper.Args()
    test_dataloader = helper.get_val_dataloader('/home/sst/dataset/nico/nico/test',args)
    net=model.PyramidNet(args.num_classes, args)
    dict=torch.load("results/kd_best6.pth")['model']
    net.network.load_state_dict(dict,strict=True)
    net.cuda()
    test(test_dataloader,net.network)




