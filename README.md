# Dependency

```
 torch
 torchvision
 torchnet
 tqdm
 Pillow
 yaml
 matplotlib
 numpy
 tensorboard
```

# Dataset

## DAVIS-like folder

A folder is said to have **DAVIS-like structure** when its directory is of the form:

```
  . root
  |- Annotations                        # annotation folder (segmentation mask)
    |- <resolution_id>                  # usually 480p
      |- <video_id>                     # ex: bear, deer, etc.
        |- <0-pad image filename>.png   # ex: 00000.png, 00051.png, etc.
  |- JPEGImages                         # image folder (RGB images)
    |- <resolution_id>                  # usually 480p
      |- <video_id>                     # ex: bear, deer, etc.
        |- <0-pad image filename>.jpg   # ex: 00000.png, 00051.png, etc.
  |- ImageSets                          # txt files for split
    |- <year>                           # ex: 2016, 2017, etc.
      |- <split name>.txt               # line-separated video name prefixes [bear, deer,...]
```

## Preprocess

At first the annotation mask will be multi-instances. If you want to split it into multiple videos (for annotations) and concurrently create symlink (for jpeg images), run

```
  python preprocess_davis.py path/to/DAVIS-like/folder
```

## Dataset class

From here on, the term **frame** is used to refer to a (JPEG image, Annotation mask) pair. 

For each input index, the **DAVISDataset** will return a pair of frame in packed form: `(support_img, support_anno, query_img), query_anno`. There are currently 3 supported modes depending on the chronological order of the support and the query frame:

0. Support and query frame do not have any particular chronological order;
1. Support is always the first frame; 
2. Support frame always comes before query frame.

## Test

To test dataset, use `test_dataset.py`:

```
usage: test_dataset.py [-h] [--root ROOT] [--imgset IMGSET] [--anno ANNO]
                       [--year YEAR] [--res RES] [--phase PHASE] [--mode MODE]

optional arguments:
  -h, --help       show this help message and exit
  --root ROOT      path to DAVIS-like folder
  --imgset IMGSET  path to ImageSet subfolder (of ROOT)
  --anno ANNO      path to Annotations subfolder (of ROOT)
  --year YEAR      path to Year subfolder (of IMGSET)
  --res RES        path to Resolution subfolder (of ANNO and JPEG)
  --phase PHASE    path to phase txt file (of IMGSET/YEAR)
  --mode MOD       mode (see Dataset class above)
```
