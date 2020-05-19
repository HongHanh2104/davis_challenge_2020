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

## Download

- DAVIS: [Homepage](https://davischallenge.org)
  - 2017: [Direct Link](https://davischallenge.org/davis2017/code.html)
- YoutubeVOS: [Homepage](https://youtube-vos.org/)
  - 2019: [Google Drive](https://drive.google.com/drive/folders/1BWzrCWyPEmBEKm0lOHe5KLuBuQxUSwqz?usp=sharing), [OneDrive](https://uillinoisedu-my.sharepoint.com/:f:/g/personal/yuchenf4_illinois_edu/Et9khbFBHEdFtGsf3ByEga0BwlRI9ONGeChm28alS4U4-w?e=9tSaGS)

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

**Note:** For the YoutubeVOS dataset, the directory needs to be modified to fit with the above form, specifically the `<resolution_id>` subfolder (for both `Annotations` and `JPEGImages`), and the entire `ImageSets` folder. A script will be prepared to automate this modification.

## Preprocess

At first the annotation mask will be multi-instances. If you want to split it into multiple videos (for annotations) and concurrently create symlink (for jpeg images), use `preprocess_davis.py`. 

```
usage: preprocess_davis.py [-h] [--root ROOT] [--copy]

optional arguments:
  -h, --help   show this help message and exit
  --root ROOT  path to DAVIS-like folder
  --copy       copy mode (create duplicate folders), default: symlink mode
               (create symbolic links)
```

**Note**: For Google Colab, it is necessary to set the `copy` flag (since symbolic links do not work on Colab yet).

## Dataset class

From here on, the term **frame** is used to refer to a (JPEG image, Annotation mask) pair. 

### DAVIS Pair Dataset

For each input index, the **DAVISPairDataset** will return a pair of frames in packed form: `(support_img, support_anno, query_img), query_anno`. There are currently 3 supported modes depending on the chronological order of the support and the query frame:

0. Support and query frame do not have any particular chronological order, i.e `(i, j)`;
1. Support is always the first frame (suitable for testing), i.e `(0, j)`; 
2. Support frame always comes before query frame, i.e `(i, j), i < j`.

There is an option to set `min_skip` and `max_skip` to bound the temporal gap between frames. There is another option `max_npairs` to bound the number of pairs sampled from each video.

There is a random counterpart **DAVISPairRandomDataset** which consists of the same set of modes, but each `getitem` call will return a sampled pair from each video. The `max_npairs` option now determines how many times the video are sampled so as to have a longer training phase.

### DAVIS Triplet Dataset

For each input index, the **DAVISTripletDataset** will return a pair of frame in packed form: `(first_img, first_anno, second_img, third_img), (second_anno, third_anno)`. There are currently 3 supported modes, in all of which, the frames are in chronological order:

0. The second frame and third frame are next to each other, i.e `(i, j, j+1)`;
1. Same as 0, but the first frame is always the first frame of the video, i.e `(0, j, j+1)` (suitable for testing);
2. Three frames are randomly separated, i.e `(i, j, k)`.

There is an option to set `min_skip` and `max_skip` to bound the temporal gap between frames. There is another option `max_npairs` to bound the number of pairs sampled from each video.

There is a random counterpart **DAVISTripletRandomDataset** which consists of the same set of modes, but each `getitem` call will return a sampled triplet from each video. The `max_npairs` option now determines how many times the video are sampled so as to have a longer training phase.

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
