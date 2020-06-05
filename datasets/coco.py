from torch.utils import data
from pycocotools.coco import COCO
from PIL import Image
import numpy as np

from pathlib import Path
import random


class COCODataset(data.Dataset):
    def __init__(self,
                 root_path=None,
                 data_type='train2017',
                 ann_folder='annotations',
                 img_folder='train2017'):
        super().__init__()

        assert root_path is not None, "Missing Missing root path, should be a path COCO dataset!"

        self.root_path = Path(root_path)

        file_json = 'instances_{}.json'.format(data_type)
        annfiles_path = self.root_path / ann_folder / file_json

        self.img_folder = self.root_path / img_folder

        # initialize COCO api for instance annotations
        self.coco = COCO(annfiles_path)

        # Get all images
        self.imgIds = sorted(self.coco.getImgIds())

    def _get_mask(self, h, w, anns):
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for i, ann in enumerate(anns):
            mask = self.coco.annToMask(ann)
            combined_mask = np.maximum(combined_mask, (i + 1) * mask)
        return combined_mask

    def __getitem__(self, inx):
        imgId = self.imgIds[inx]
        imgId = self.coco.loadImgs(imgId)[0]

        #response = requests.get(imgId['coco_url'])
        #img_bytes = io.BytesIO(response.content)
        path = self.img_folder / imgId['file_name']
        img = Image.open(path).convert('RGB')
        # Get all annotations of this image
        annIds = self.coco.getAnnIds(imgIds=imgId['id'], iscrowd=False)

        # Choose randomly 3 in annIds
        annIds = random.sample(annIds, k=min(3, len(annIds)))

        anns = self.coco.loadAnns(annIds)
        mask = self._get_mask(imgId['height'], imgId['width'], anns)
        mask = Image.fromarray(mask).convert('P')

        return img, mask

    def __len__(self):
        return len(self.imgIds)
