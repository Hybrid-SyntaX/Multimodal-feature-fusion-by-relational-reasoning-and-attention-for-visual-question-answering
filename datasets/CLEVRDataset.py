import json
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset  # For custom datasets


# import torchvision.transforms.functional as F @BUG : causes signal kill for some reason
#
# {"split": "train",
#  "image_filename":
#      "CLEVR_train_001429.png",
#  "answer": "yes", "question": "Is there a blue cylinder?",
#  "image_index": 1429},


class CLEVRDataset(Dataset):
    def __init__(self, questions_filepath, images_dir, image_transform=None, use_cached_image_features=False,
                 blind=False
                 , split_percentage=None, split_type='random'
                 ):
        self.images_dir = images_dir
        self.image_transform = image_transform
        self.use_cached_image_features = use_cached_image_features
        # self.get_question_family_index=get_question_family_index
        with open(questions_filepath, 'r') as questions_file:
            self.questions = json.load(questions_file)['questions']

        # if split_percentage and split_type:
        #     self.questions=split_by_question_type(self.questions, percentage = split_percentage,split_type = split_type)

        self.blind = blind
        # all_questions = self.questions  # 149991
        # sub_len = len(self.questions) // 30
        # self.questions = all_questions[:sub_len]  # 14* 10000
        # self.image_feats=[]
        # for index in tqdm(range(len(self.questions)),desc = f'Loading images into memory from {self.images_dir}'):
        #     #image=np.load(os.path.join(self.images_dir, str(self.questions[index]['image_index'])) + '.npz')['x']
        #     #image_filename = os.path.join(self.images_dir, self.questions[index]['image_filename'])
        #     #image = rgba2rgb(io.imread(image_filename)).astype(float)# / 255
        #
        #     name,ext=os.path.splitext( self.questions[index]['image_filename'])
        #     image=np.load(os.path.join(self.images_dir,name+'.npy')).astype(float)
        #     image = torch.from_numpy(image).float()
        #     self.image_feats.append(image)

        # self.use_h5=False
        # if self.use_h5:
        #     self.h5_file = h5py.File(images_dir+ '.h5','r')['features']
        #     with open(images_dir + '.json','r') as file:
        #         self.img2idx=json.load(file)

    def __getitem__(self, index):
        question = self.questions[index].get('question')
        answer = self.questions[index].get('answer')
        question_family_index = self.questions[index].get('question_family_index')

        if self.blind:
            image = None
        elif self.use_cached_image_features:
            # if self.use_h5:
            #     image_idx = str(self.questions[index]['image_index'])
            #     h5_idx=self.img2idx[image_idx]
            #     image = torch.HalfTensor(self.h5_file[h5_idx])
            # else:

            image = np.load(os.path.join(self.images_dir, str(self.questions[index]['image_index'])) + '.npz')['x']
            # image = self.image_feats[index]
        else:
            image_filename = os.path.join(self.images_dir, self.questions[index]['image_filename'])

            image = Image.open(image_filename).convert('RGB')

            # image= Image.open(image_filename).convert('RGBA')

            # image = Image.new('RGBA', png.size, (255, 255, 255))

            # image.load()
            #

            # image = rgba2rgb(io.imread(image_filename)) #/ 255

            # another feature  set
            # image = self.get_image(index)
            # image = self.image_feats[index]
            # image = np.moveaxis(image, -1, 0).astype(float)
            # image=None
        if self.image_transform and not self.blind:
            # image = F.to_pil_image(image)

            image = self.image_transform(image)
            # image = image.transpose(0,1)

        return image, question, answer, question_family_index

    def get_image(self, index):
        name, ext = os.path.splitext(self.questions[index]['image_filename'])
        image = np.load(os.path.join(self.images_dir, name + '.npy')).astype(float)
        image = torch.from_numpy(image).float()
        return image

    def __len__(self):
        return len(self.questions)
