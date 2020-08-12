# Requirements

This project was done and tested using Python 3.8 so it should work for Python 3.8+.

To create a conda virtual environment that would work, run

```
  conda env create -f environment.yaml
```

To install the required packages using pip, run

```
  pip install -r requirements.txt
```

# Usage

## Train

### Train

To train, run
```
  python train.py --config path/to/config/file [--gpus gpu_id] [--debug]
```

Arguments:
```
  --config: path to configuration file
  --gpus: gpu id to be used
  --debug: to save the weights or not
```

For example:
```
  python train.py --config configs/train/stm_fss.yaml --gpus 0 --debug
```

### Config

Modify the default configuration file (YAML format) to suit your need, the properties' name should be self-explanatory.

### Result

All the result will be stored in the ```runs``` folder in separate subfolders, one for each run. The result consists of the log file for Tensorboard, the network pretrained models (best metrics, best loss, and the latest iteration).

#### Training graph

This project uses Tensorboard to plot training graph. To see it, run

```
  tensorboard --logdir=logs
```

and access using the announced port (default is 6006, e.g ```http://localhost:6006```).

#### Pretrained models

The ```.pth``` files contains a dictionary:

```
  {
      'epoch':                the epoch of the training where the weight is saved
      'model_state_dict':     model state dict (use model.load_state_dict to load)
      'optimizer_state_dict': optimizer state dict (use opt.load_state_dict to load)
      'log':                  full logs of that run
      'config':               full configuration of that run
  }
```

## Eval

To test a pretrained model, run
```
  python eval.py -w path/to/pretrained/model [-g 0] [-f 0] [-s 3698]
```

Arguments:
```
  -w: path to pretrained model
  -g: gpu id to be used (default: 0)
  -f: fold to evaluate (default: 0)
  -s: random seed (default: 3698)
```

For example:
```
  python eval.py -w runs/test_best_acc.pth -g 1 -f 1 -s 1234
```

## Visualize

To make inference on one sample, run
```
  python visualize.py --weight path/to/pretrained/model --ref_img path/to/reference/image --ref_mask path/to/reference/mask --query_img path/to/query/image
```