
### NICO

## How to run ?

please download NICO dataset and recommend enviroment to meet the requirement (see requirment.txt)

```bash
conda create -n torch python==3.9.0

activate torch

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install timm numpy Pillow tqdm

python total_train.py --fp16 --parallel --cuda_devices '0,1,2,3,4,5,6,7' --train_image_path \
<you training dataset path> --label2id_path <you json path> --test_image_path <you test dataset path>
```

## How to test?

```bash
mv <you pretrain ckpt file> resume.pth
python train.py --test --if_resume --train_image_path \
<you training dataset path> --label2id_path <you json path> --test_image_path <you test dataset path>
```