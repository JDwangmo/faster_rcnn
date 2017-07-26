import torch
import torch.utils.data as data
from PIL import Image, ImageDraw
import os
import os.path
import sys
import torchvision.transforms as transforms



class POSDetection(data.Dataset):
    def __init__(self, root, image_set, transform=None, target_transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        dataset_name = 'VOC2007'
        self._annopath = os.path.join(self.root, 'posGt', '%s.txt')
        self._imgpath = os.path.join(self.root, 'pos', '%s.png')
        self._imgsetpath = os.path.join(self.root, 'ImageSets', 'Main', '%s.txt')

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()

        img = Image.open(self._imgpath % img_id).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def show(self, index):
        img, target = self.__getitem__(index)
        draw = ImageDraw.Draw(img)
        for obj in target['boxes']:
            draw.rectangle(obj.numpy().tolist(), outline=(255, 0, 0))
            # draw.text(obj[0:2].numpy().tolist(), obj[3], fill=(0, 255, 0))
        img.show()


if __name__ == '__main__':
    cls = ('__background__',  # always index 0
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
    class_to_ind = dict(zip(cls, range(len(cls))))

    ds = POSDetection('/home/jdwang/PycharmProjects/hand_detection/', 'train',
                      # transform=transforms.ToTensor(),
                      )
    print(len(ds))
    img, target = ds[0]
    print(target)
    # print(img.size())
    # import numpy as np
    # print(np.asarray(img).shape)
    # print(img.show())
    ds.show(1)
    # dss = VOCSegmentation('/home/francisco/work/datasets/VOCdevkit/', 'train')
    # img, target = dss[0]

    # img.show()
    # print(target_transform(target))