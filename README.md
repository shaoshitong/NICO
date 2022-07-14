
### NICO

---

## How to run ?

please download NICO dataset and recommend enviroment to meet the requirement (see requirment.txt)

```bash
conda env create -f environment.yaml
python total_train.py --fp16 --parallel --cuda_devices '0,1,2,3,4,5,6,7' --train_image_path --track_mode <track1 or track2>\
<you training dataset path> --label2id_path <you json path> --test_image_path <you test dataset path>
```
---

## How to test?

```bash
mv <you pretrain ckpt file> resume.pth
python train.py --test --if_resume --train_image_path --track_mode <track1 or track2>\
<you training dataset path> --label2id_path <you json path> --test_image_path <you test dataset path>
``` 

---

## How to train track1 dataset immediately?

# 1. Download the track1 dataset and finish unpacking it, then place it in the root track1 folder


make ensure that the files are placed as follows:
```bash
+-NICO/
  |
  *-dg_label_id_mapping.json
  |
  +-nico/
    |
    +-test/
    |
    +-train/
      |
      +-rock/
      | |
      | +-hot air balloon/
      | |
      | +-goose/
      | |
      | +-frog/
      | |
      | +-mailbox/
      | |
      | +-bus/
      | |
      | +-shrimp/
      | |
      | +-airplane/
      | |
      | +-crocodile/
      | |
      | +-pineapple/
      | |
      | +-cow/
      | |
      | +-tortoise/
      | |
      | +-sheep/
      | |
      | +-scooter/
      | |
      | +-lion/
      | |
      | +-seal/
      | |
      | +-dolphin/
      | |
      | +-pumpkin/
      | |
      | +-racket/
      | |
      | +-fox/
      | |
      | +-sunflower/
      | |
      | +-car/
      | |
      | +-corn/
      | |
      | +-elephant/
      | |
      | +-sailboat/
      | |
      | +-dog/
      | |
      | +-tent/
      | |
      | +-flower/
      | |
      | +-football/
      | |
      | +-hat/
      | |
      | +-chair/
      | |
      | +-cat/
      | |
      | +-owl/
      | |
      | +-cactus/
      | |
      | +-fishing rod/
      | |
      | +-ship/
      | |
      | +-clock/
      | |
      | +-wheat/
      | |
      | +-spider/
      | |
      | +-umbrella/
      | |
      | +-horse/
      | |
      | +-ostrich/
      | |
      | +-giraffe/
      | |
      | +-wolf/
      | |
      | +-helicopter/
      | |
      | +-kangaroo/
      | |
      | +-bicycle/
      | |
      | +-bird/
      | |
      | +-butterfly/
      | |
      | +-motorcycle/
      | |
      | +-monkey/
      | |
      | +-rabbit/
      | |
      | +-crab/
      | |
      | +-squirrel/
      | |
      | +-bear/
      | |
      | +-train/
      | |
      | +-tiger/
      | |
      | +-lifeboat/
      | |
      | +-lizard/
      | |
      | +-truck/
      | |
      | +-gun/
      |
      +-outdoor/
      | |
      | +-hot air balloon/
      | |
      | +
      | .
      | .
      +-autumn/
      | |
      | +-hot air balloon/
      | |
      | +
      | .
      | .
      +-dim/
      | |
      | +-hot air balloon/
      | |
      | +
      | .
      | .
      +-water/
        |
        +-hot air balloon/
        |
        +
        .
        .
```

# 2. Create your environment for training
```bash
conda env create -f environment.yaml
source activate nico-mcislab840 # in [Linux] , activate nico-mcislab840 # in [Window]
```
# 3. Modify train_image_path, label2id_path, and test_image_path in track1_run.sh

# 4. Run track1_run.sh
```bash
chmod 777 track1_run.sh
bash track1_run.sh
```

---


# 5. Output the test csv file

```bash
python train.py --test --batchsize 64 --track_mode track1 \
        --train_image_path /home/Bigdata/NICO/nico/train/ --label2id_path \
        /home/Bigdata/NICO/dg_label_id_mapping.json --test_image_path /home/Bigdata/NICO/nico/test/
```

## How to train track2 dataset immediately?

# 1. Download the track2 dataset and finish unpacking it, then place it in the root track2 folder

make ensure that the files are placed as follows:

```bash
+-NICO2/
  |
  +-ood_label_id_mapping.json
  |
  +-nico/
    |
    +-test/
    |
    +-train/
      |
      +-hot air balloon/
      |
      +-snake/
      |
      +-lemon/
      |
      +-mushroom/
      |
      +-chicken/
      |
      +-mailbox/
      |
      +-bus/
      |
      +-airplane/
      |
      +-beetle/
      |
      +-cow/
      |
      +-tortoise/
      |
      +-sheep/
      |
      +-tank/
      |
      +-lion/
      |
      +-monitor/
      |
      +-seal/
      |
      +-dolphin/
      |
      +-pumpkin/
      |
      +-fox/
      |
      +-sunflower/
      |
      +-car/
      |
      +-bee/
      |
      +-elephant/
      |
      +-sailboat/
      |
      +-dog/
      |
      +-flower/
      |
      +-camera/
      |
      +-hat/
      |
      +-chair/
      |
      +-cat/
      |
      +-cactus/
      |
      +-fishing rod/
      |
      +-ship/
      |
      +-wheat/
      |
      +-pepper/
      |
      +-spider/
      |
      +-umbrella/
      |
      +-horse/
      |
      +-helicopter/
      |
      +-fish/
      |
      +-bicycle/
      |
      +-phone/
      |
      +-bird/
      |
      +-motorcycle/
      |
      +-monkey/
      |
      +-rabbit/
      |
      +-squirrel/
      |
      +-bear/
      |
      +-sword/
      |
      +-cauliflower/
      |
      +-train/
      |
      +-whale/
      |
      +-shark/
      |
      +-banana/
      |
      +-penguin/
      |
      +-lifeboat/
      |
      +-camel/
      |
      +-truck/
      |
      +-gun/
      |
      +-dragonfly/
```
# 2. Create your environment for training
```bash
conda env create -f environment.yaml
source activate nico-mcislab840 # in [Linux] , activate nico-mcislab840 # in [Window]
```
# 3. Modify train_image_path, label2id_path, and test_image_path in track2_run.sh

# 4. Run track2_run.sh
```bash
chmod 777 track2_run.sh
bash track2_run.sh
```

# 5. Output the test csv file

```bash
python train.py --test --batchsize 64 --track_mode track2 \
        --train_image_path /home/Bigdata/NICO2/nico/train/ --label2id_path \
        /home/Bigdata/NICO2/ood_label_id_mapping.json --test_image_path /home/Bigdata/NICO2/nico/test/
```