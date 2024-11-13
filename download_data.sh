#!/bin/bash

# set the destination
destination="nsd_data"

subdirs=("train" "test" "val")

for subdir in "${subdirs[@]}"; do
    full_destination="${destination}/webdataset_avg_split/${subdir}/"
    mkdir -p "$full_destination"
done

declare -a i_values=(1 2 5 7)

# Download the train set
for i in "${i_values[@]}"; do
  for j in {0..17}; do
    url="https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/main/webdataset_avg_split/train/train_subj0${i}_${j}.tar"
    wget -P "${destination}/webdataset_avg_split/train" "$url"
  done
done

# Download the validation set
for i in "${i_values[@]}"; do
  url="https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/main/webdataset_avg_split/val/val_subj0${i}_0.tar"
  wget -P "${destination}/webdataset_avg_split/val" "$url"
done

# Download the test set
for i in "${i_values[@]}"; do
  for j in {0..1}; do
    url="https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/main/webdataset_avg_split/test/test_subj0${i}_${j}.tar"
    wget -P "${destination}/webdataset_avg_split/test" "$url"
  done
done
