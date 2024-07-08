This readme are mostly follows the the same Guide in [ProGrad](https://github.com/BeierZhu/Prompt-align).

# How to Run

## GPU memory needed

All the experiments is able to run on a single graphic card. However, **if you want to get results on ImageNet, the memory on any single graphic card should be larger than 24 GB.** Around 12 GB is enough for other datasets. 


## How to Install
This code is built on top of the toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). But we have some modification on it. So please install the provided Dassl.provp.pytorch. Go the the folder Dassl.provp.pytorch provided in the appendix, and prepare the environment as follows:

```
# Create a conda environment
conda create -n dassl python=3.7

# Activate the environment
conda activate dassl

# Install dependencies
pip install -r requirements.txt

# Install torch (version >= 1.7.1) and torchvision
# Please make sure you have installed the gpu version due to the speed.
# For example:
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```

After that, run `pip install -r requirements.txt` under `provp.public/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated). Then, you are ready to go.

Follow [DATASETS.md](DATASETS.md) to install the datasets.

## Few-shot setting on 11 datasets

Basic format:
```
bash provp.sh ${DATASET_NAME} ${NUMBER_OF_PROMPTS}${CONFIG_NAME} ${SHOTS}
```

For example, to run 1, 2, 4, 8, and 16 shots on stanford_cars, 

- 1 shot: `bash provp.sh stanford_cars 50 vit_b16_ep50 1 `
- 2 shots: `bash provp.sh stanford_cars 50 vit_b16_ep100  2 `
- 4 shots: `bash provp.sh stanford_cars 50 vit_b16_ep100  4`
- 8 shots: `bash provp.sh stanford_cars 50  vit_b16  8`
- 16 shots: `bash provp.sh stanford_cars 50 vit_b16  16 `

You can change provp.sh to promp.sh and remove the **NUMBER_OF_PROMPTS** config to try the multi-modal version .


```
output
|–– caltech101/
|   |–– CoOp/
|   |   |–– vit_b16_16shots/
|   |   |   |–– seed1/
|   |   |   |–– seed2/
|   |   |   |–– seed3/
|   |   |–– vit_b16_8shots/
|   |   |   |–– seed1/
|   |   |   |–– seed2/
|   |   |   |–– seed3/
```

To calculate the average results for the folder `vit_b16_16shots/`, you can run

```bash
python parse_test_res.py output/caltech101/CoOp/vit_b16_16shots/
```

Then, you will see something like this in your terminal

```bash
===
Summary of directory: output/caltech101/CoOp/vit_b16_16shots/***
* accuracy: xx.00% +- 0.xx%
* error: x.00% +- 0.xx%
===
```

## Zero-Shot CLIP
See `CoOp/scripts/zeroshot.sh`.

## Generalization From Base to New Classes

You will need `base2new_train_provp.sh`, `base2new_test_provp.sh`, `base2new_train_provp.sh`, and `base2new_test_provp.sh`. The scripts with the prefix `base2new_train` train a model on base classes while the ones with the prefix `base2new_test` evaluate the trained model on new classes. Both kinds of scripts have only one input argument, i.e., `DATASET`. `DATASET` takes as input a dataset name, like `imagenet` or `caltech101`. The valid names are the files' names in `CoOp/configs/datasets/`.

The scripts with postfix `provp.sh` are used for our proposed method. We also support the pioneer works as `coop.sh` or `prograd.sh`.

Below we provide an example on how to evaluate the model on ImageNet.
```
bash provp.sh ${DATASET_NAME} ${NUMBER_OF_PROMPTS}
```
as 
```bash
bash base2new_train_provp.sh stanford_cars 16
bash base2new_test_provp.sh stanford_cars 16
```

When the evaluation is done, you can use `parse_test_res.py` to automatically calculate the average results. For instance, after you finish the evaluation using the aforementioned commands, you would get

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– stanford_cars/
|   |   |   |–– shots_16/
|   |   |   |   |–– ProVP/
|   |   |   |   |   |–– vit_b16_ep100/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– stanford_cars/
|   |   |   |–– shots_16/
|   |   |   |   |–– ProVP/
|   |   |   |   |   |–– vit_b16_ep100/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```

Then, to get the average performance on the base classes, run

```bash
python parse_test_res.py output/base2new/train_base/stanford_cars/shots_16/ProVP/vit_b16_ep100
```

To get the average performance on the new classes, run

```bash
python parse_test_res.py output/base2new/test_new/stanford_cars/shots_16/ProVP/vit_b16_ep100 --test-log
```

