import json
import os
from glob import glob

import numpy as np
from skimage import io
import torch
from PIL import Image
from skimage.color import rgba2rgb
from torch.utils.data.dataset import Dataset  # For custom datasets

class COCOQADataset(Dataset):
    def __init__(self, questions_dir, images_dir, image_transform=None):
        self.images_dir = images_dir
        self.image_transform=image_transform

        self.questions= self.read_file(os.path.join(questions_dir, 'questions.txt'))
        self.answers = self.read_file(os.path.join(questions_dir, 'answers.txt'))
        self.types = self.read_file(os.path.join(questions_dir, 'types.txt'))
        self.image_ids = self.read_file(os.path.join(questions_dir, 'img_ids.txt'))


    @staticmethod
    def read_file( filepath):
        with open(os.path.join(filepath), 'r') as f:
            return f.readlines()

    def __getitem__(self, index):
        question = self.questions[index].strip()
        answer = self.answers[index].strip()
        type = self.types[index].strip()
        iamge_id = self.image_ids[index].strip()


        image_filename=glob(os.path.join(self.images_dir, f'*{iamge_id}.jpg'))[0]
        image = io.imread(image_filename)/255
        image = np.moveaxis(image, -1, 0)

        if self.image_transform is not None:
            image_tensor = self.image_transform(image)


        return  image,question,answer

    def __len__(self):
        return len(self.questions)
