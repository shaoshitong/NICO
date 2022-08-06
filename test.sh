#!/bin/bash

label="kk"
test_pth_path=( "original_1.pth" "original_2.pth"  "original_3.pth" )
track_mode="track1"
label2id_path="/home/Bigdata/NICO/dg_label_id_mapping.json"
test_image_path="/home/Bigdata/NICO/nico/test/"

save_json_path="prediction$label"
mkdir $save_json_path
p="/"
for i in ${test_pth_path[*]}; do
  json_save_path="$label${i/.pth/}prediction.json"
  python test.py --test_pth_path $i --track_mode $track_mode --json_save_path $json_save_path \
          --label2id_path $label2id_path --test_image_path $test_image_path
  mv $json_save_path "$save_json_path$p"
done

final="final_prediction.json"
python ensemble.py --ensemble_path $save_json_path --save_path "$label$final"