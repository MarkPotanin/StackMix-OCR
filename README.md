# StackMix and Blot Augmentations for Handwritten Recognition using CTCLoss

[This paper](https://arxiv.org/abs/2108.11667) presents a new text generation method StackMix. StackMix can be 
applied to the standalone task of generating handwritten text based on printed text.

## Table with results

[TODO] 

## Demo Neptune with all experiments

[TODO]

## Pretrained Models

[TODO] 

## Recommended structure of experiments:

```
--- StackMix-OCR/
--- StackMix-OCR-DATA/
--- StackMix-OCR-SAVED_MODELS/
--- StackMix-OCR-MWE_TOKENS/
```
## Config file
Create a new config file need  in ```configs/init.py```. 
An individual config file is required for each dataset

## Datasets

### There are two ways to get a dataset:
#### The first way:
1. Download selected dataset and annotations from original site (for example Bentham: http://www.transcriptorium.eu/~tsdata/BenthamR0/)  
2. Prepare dataset using jupyter notebook in jupyters folder  
3. Put dataset in StackMix-OCR-DATA folder  

####  The second way:
Downlad prepared dataset using our script ```download_dataset.py``` (for example Bentham: ```python scripts/download_dataset.py --dataset_name=bentham```) 
And now you can use train script.  

You can change out folder by key --data_dir='your path', by default --data_dir=../StackMix-OCR-DATA.
All dataset names: bentham, peter, hkr, iam.

### Dataset format

The dataset should contain a directory with images, a csv file and a json file with markup information. An example of the file location can be seen below

![Dataset format](https://sun9-7.userapi.com/impg/GpIzvjYF9AbpGOQbamvCcgwRA9fVfHo2SaPOcg/Ox847-h0m8o.jpg?size=174x106&quality=96&sign=fb2ce9af30b54f09cfc8542ee8f84fad&type=album)

The file with the csv extension must contain a separate field with information about which selection the image belongs to `(train / val / test)`.
Example of the structure and content of a csv file is given below
```
sample_id,path,stage,text
270-01,washington/images/270-01.png,train,"270. Letters, Orders and Instructions. October 1755."
270-03,washington/images/270-03.png,train,"only for the publick use, unless by particu-"
270-04,washington/images/270-04.png,train,lar Orders from me. You are to send
270-05,washington/images/270-05.png,train,"down a Barrel of Flints with the Arms, to"
```

#### How to get char masks:
```
python scripts/prepare_char_masks.py \
--checkpoint_path "exp/hkr_base_no_aug/last.pt" \
--dataset_name "peter" \
--image_w 1024 \
--image_h 128 \
--bs 12 \
--num_workers 3 \
--experiment_name "sdfsdf" \
--data_dir "data/"
```

## Run in docker:

[TODO] 

## Run locally:

install requirements:
```
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

### Run experiments from root project (StackMix-OCR):

1. Train "base" experiment:
```
sh runners/bentham/train.sh
```

2. Train "base + blots" experiment:
```
sh runners/bentham/train_blots.sh
```

3. Train "base + stackmix" experiment:
```
sh runners/bentham/prepare_stackmix.sh
sh runners/bentham/train_stackmix.sh
```

4. Train "base + blots + stackmix" experiment:
```
sh runners/bentham/prepare_stackmix.sh
sh runners/bentham/train_blots_stackmix.sh
```

## Example run train:
```
python scripts/run_train.py \
--checkpoint_path "" \
--experiment_name "HKR_base_aug" \
--dataset_name "hkr" \
--data_dir "data/" \
--output_dir "exp/" \
--experiment_description \
"[Base] Training Base OCR on HKR dataset" \
--image_w 256 \
--image_h 32 \
--num_epochs 300 \
--bs 64 \
--num_workers 6 \
--use_blot 0 \
--use_augs 1 \
--use_progress_bar 0 \
--seed 6955
```

## Example run evaluation:
```
python scripts/run_evaluation.py \
--experiment_folder "exp/htr_dataset_aug" \
--dataset_name "htr_dataset" \
--data_dir "data/" \
--image_w 1024 \
--image_h 128 \
--bs 64 \
--seed 6955
```

## Generating char_mask for stackmix
```
python scripts/prepare_char_masks.py \
--checkpoint_path "exp/hkr_base_no_aug/last.pt" \
--dataset_name "peter" \
--image_w 1024 \
--image_h 128 \
--bs 12 \
--num_workers 3 \
--experiment_name "sdfsdf" \
--data_dir "data/"
```

## Generating images with stackmix

[TODO] scripts

Example of generating images with stackmix

![Example of generating images](https://sun9-64.userapi.com/impg/xAFmDnVuuTmc4FM_FKhLPnq-KvrppD4x-DvUKg/hy1qKbRbS58.jpg?size=402x305&quality=96&sign=5bdfa7702f2e655cc991e274d4bb7b3f&type=album)

## Supported by:

- Sber
- OCRV
- Sirius University
- RZHD


## Citation

Please cite the related works in your publications if it helps your research:

[TODO]

## Contacts

- [A. Shonenkov](https://www.kaggle.com/shonenkov) shonenkov@phystech.edu
- [D. Karachev](https://github.com/thedenk/)
- [M. Novopoltsev](https://github.com/maximazzik)
- [D. Dimitrov]
- [M. Potanin](https://github.com/MarkPotanin)
