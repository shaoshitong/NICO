
# NICO

---

## The most important thing you should pay attention to is:

<font color="red"> Make sure the last character of the root directory of all image folder's path is '\'</font>

<font color="red"> Make sure the student.pth file exists before outputting the json file </font>

---

## How to train track1 dataset immediately?

### 1. Download the track1 dataset and finish unpacking it, then place it in the root track1 folder


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

### 2. Create your environment for training
```bash
conda env create -f environment.yaml
source activate nico-mcislab840 # in [Linux] , activate nico-mcislab840 # in [Window]
```
### 3. Modify train_image_path, label2id_path, and test_image_path in track1_run.sh and ensemble_track1_run.sh

### 4. Run ensemble_track1_run.sh for ensemble, and then get the final json => final_prediction.json

```bash
chmod 777 ensemble_track1_run.sh
bash ensemble_track1_run.sh
```

## How to train track2 dataset immediately?

### 1. Download the track2 dataset and finish unpacking it, then place it in the root track2 folder

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
### 2. Create your environment for training
```bash
conda env create -f environment.yaml
source activate nico-mcislab840 # in [Linux] , activate nico-mcislab840 # in [Window]
```
### 3. Modify train_image_path, label2id_path, and test_image_path in track2_run.sh and ensemble_track2_run.sh


### 4. Run ensemble_track2_run.sh for ensemble, and then get the final json => final_prediction.json

```bash
chmod 777 ensemble_track2_run.sh
bash ensemble_track2_run.sh
```
