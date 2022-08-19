
# NICO

---

## Introduction

🤗 We achieved third place 🏆 in the 2022 NICO Common Context Generalization Challenge (ECCV 2022 Workshop), and the related code will be released here.

## Datasets:

(NICO++)[https://arxiv.org/abs/2204.08040]

### downloads:

The released data (for NICO challenge) is available:

[dropbox](https://www.dropbox.com/sh/u2bq2xo8sbax4pr/AADbhZJAy0AAbap76cg_XkAfa?dl=0)

[Tsinghua Cloud](https://www.dropbox.com/sh/u2bq2xo8sbax4pr/AADbhZJAy0AAbap76cg_XkAfa?dl=0)

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

You can also free to use NICO++ data for your research for non-economic purpose.

### The most important thing you should pay attention to is:

<font color="red"> Make sure the last character of the root directory of all image folder's path is '\'</font>


## How to train track1 dataset immediately?

### 1. Create your environment for training
```bash
conda env create -f environment.yaml
source activate nico-mcislab840 # in [Linux] , activate nico-mcislab840 # in [Window]
```
### 2. Modify train_image_path, label2id_path, and test_image_path in track1_run.sh

### 3. Run ensemble_track1_run.sh for ensemble, and then get three final checkpoints: `track_1_pth_1.pth`, `track_1_pth_2.pth`, `track_1_pth_3.pth`

```bash
chmod 777 ensemble_track1_run.sh
bash ensemble_track1_run.sh
```

## How to test track1 dataset immediately?

### 1. Modify test_image_path, label2id_path, and test_pth_path in test.sh

### 2. Run test.sh for Test Time Augmentation (a longer period of time is required).

```bash
chmod 777 test.sh
bash test.sh
```

### 3. The voting method is applied for the final prediction (final_prediction.json)
```bash
python ensemble.py --ensemble_path predictionnico1 --save_path final_prediction.json
```