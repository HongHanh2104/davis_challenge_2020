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

At first the annotation mask will be multi-instances. If you want to split it into multiple videos (for annotations) and concurrently create symlink (for jpeg images), use `preprocess_davis.py`. For Google Colab, it is necessary to set the `copy` flag.

```
usage: preprocess_davis.py [-h] [--root ROOT] [--copy]

optional arguments:
  -h, --help   show this help message and exit
  --root ROOT  path to DAVIS-like folder
  --copy       copy mode (create duplicate folders), default: symlink mode
               (create symbolic links)
```

## Dataset class

From here on, the term **frame** is used to refer to a (JPEG image, Annotation mask) pair. 

### DAVIS Pair Dataset

For each input index, the **DAVISPairDataset** will return a pair of frame in packed form: `(support_img, support_anno, query_img), query_anno`. There are currently 3 supported modes depending on the chronological order of the support and the query frame:

0. Support and query frame do not have any particular chronological order;
1. Support is always the first frame (suitable for testing); 
2. Support frame always comes before query frame.

### DAVIS Triplet Dataset

For each input index, the **DAVISTripletDataset** will return a pair of frame in packed form: `(first_img, first_anno, second_img, third_img), (second_anno, third_anno)`. There are currently 2 supported modes, in both of which, the frames are in chronological order:

0. Second and third frame are next to each other, in other words, `(i, j, j+1)`
1. Three frames are randomly separated.

There is an option to set `max_skip` to prevent getting frames that are too far apart.

## Test

To test dataset, use `test_dataset.py`:

```
usage: test_dataset.py [-h] [--root ROOT] [--anno ANNO] [--jpeg JPEG]
                       [--res RES] [--imgset IMGSET] [--year YEAR]
                       [--phase PHASE] [--mode MODE]

optional arguments:
  -h, --help       show this help message and exit
  --root ROOT      path to DAVIS-like folder
  --anno ANNO      path to Annotations subfolder (of ROOT)
  --jpeg JPEG      path to JPEGImages subfolder (of ROOT)
  --res RES        path to Resolution subfolder (of ANNO and JPEG)
  --imgset IMGSET  path to ImageSet subfolder (of ROOT)
  --year YEAR      path to Year subfolder (of IMGSET)
  --phase PHASE    path to phase txt file (of IMGSET/YEAR)
  --mode MODE      frame pair selector mode
```
