#!/bin/bash

wget https://raw.githubusercontent.com/ardisdataset/ARDIS/master/ARDIS_DATASET_IV.rar
# mkdir data/ARDIS
unrar e ARDIS_DATASET_IV.rar data/ARDIS/
rm -rf ARDIS_DATASET_IV.rar
