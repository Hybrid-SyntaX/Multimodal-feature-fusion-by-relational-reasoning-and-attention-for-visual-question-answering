import json
import os
from glob import glob

import h5py
from torch.utils.data.dataset import Dataset  # For custom datasets


class GQADataset(Dataset):
    def __init__(self, questions_filepath, spatial_features_path=None, object_features_path=None, transform=None, blind=False):
        # (questions, spatial_features)
        self.spatial_features_path = spatial_features_path
        self.object_features_path=object_features_path
        #self.subset = questions_path.split('/')[-1].replace('_questions.json', '')
        self.subset= os.path.basename(questions_filepath).replace('_questions.json', '').replace('trimmed_', '')
        self.blind=blind
        with open(questions_filepath, 'r') as questions_json_file:
            questions_json = json.load(questions_json_file)
        if  spatial_features_path:
            self.use_spatial_features=True
        else:
            self.use_spatial_features = False

        self.questions = list(questions_json.values())

        if spatial_features_path:
            spatial_features_json_path = os.path.join(spatial_features_path, self.subset+'_gqa_spatial_info.json')

            with open(spatial_features_json_path,'r') as spatial_features_json_file:
                self.spatial_info_json = json.load(spatial_features_json_file)

        if object_features_path:
            object_features_json_path = os.path.join(object_features_path, self.subset+'_gqa_objects_info.json')

            with open(object_features_json_path,'r') as object_features_json_file:
                self.object_info_json = json.load(object_features_json_file)


        if not blind:
            if spatial_features_path:
                spatial_features_h5=glob(os.path.join(spatial_features_path,'*.h5'))
                self.spatial_features=self.load_spatial_features(spatial_features_h5)
            if object_features_path:
                objects_features_h5 = glob(os.path.join(object_features_path, '*.h5'))
                self.object_features = self.load_object_features(objects_features_h5)


        #self.spatial_features=spatial_features_h5['features']

        #self.labels=self.data.get('label')
        #self.labels=np.array(self.data.get('label')).astype(torch.LongTensor)
        
        #self.height = 7 #self.spatial_features.shape[2] #7
        #self.width = 7 #self.spatial_features.shape[3] #7
        self.image_transform = transform


    @staticmethod
    def load_spatial_features(spatial_features_h5):
        img_features = {}
        for h5_filepath in spatial_features_h5:
            # print(h5_path)
            key = int(os.path.basename(h5_filepath).split('_')[-1].split('.')[0])

            img_features[key] = h5py.File(h5_filepath,'r')['features']
        return img_features
    @staticmethod
    def load_object_features(object_features_h5):
        img_features = {}
        for h5_filepath in object_features_h5:
            # print(h5_path)
            key = int(os.path.basename(h5_filepath).split('_')[-1].split('.')[0])
            h5_file=h5py.File(h5_filepath,'r')
            img_features[key] = {'features':h5_file['features'],
                                 'bboxes':h5_file['bboxes']}
            #img_features[key] = h5_file['bboxes']
        return img_features
    def __getitem__(self, index):
        question = self.questions[index]['question']
        answer = self.questions[index]['answer']
        fullAnswer = self.questions[index]['fullAnswer']
        imageId = self.questions[index]['imageId']
        if not self.blind:

            #Reading spatial feature from corrosponding h5 file
            if self.spatial_features_path:
                spatial_feature,spatial_metadata = self.get_spatial_feature(imageId)
            else:
                spatial_metadata=None
                spatial_feature=None
            #object_height=self.get_object_feature(imageId,'height')
            #object_width = self.get_object_feature(imageId, 'width')
            #object_height=self.object_info_json[imageId]['height']
            #object_width=self.object_info_json[imageId]['width']
            if self.object_features_path:
                bbox,object_metadata=self.get_object_feature(imageId,'bboxes')
                object_feats,_=self.get_object_feature(imageId, 'features')
            else:
                object_metadata = None
                object_feats = None
                bbox=None

            if self.spatial_features_path:
                metadata={
                    'imageId':imageId,
                    'spatial':spatial_metadata,
                    'object':object_metadata
                }
            else:
                metadata = {
                    'imageId': imageId,
                    'spatial': spatial_metadata,
                    'object': object_metadata
                }
                #spatial_feature=None

            if self.image_transform is not None:
                feature_as_tensor = self.image_transform(spatial_feature)
        else:
            #spatial_feature=torch.randn((2048,7,7))
            #object_feats=torch.randn((100,2048))
            #bbox=torch.randn((100,4))
            spatial_feature=None
            object_feats=None
            bbox=None
            if self.use_spatial_features:
                spatial_metadata= self.spatial_info_json[imageId]
            else:
                spatial_metadata=None
            metadata = {
                'imageId': imageId,
                'spatial': spatial_metadata,
                'object': self.object_info_json[imageId]
            }
        # Return image and the label
        return spatial_feature,object_feats,bbox, question, answer, fullAnswer,metadata  #single_image_label



    def __len__(self):
        return len(self.questions)

    def get_spatial_feature(self, imageId):
        spatial_metadata = self.spatial_info_json[imageId]
        spatial_feature_idx = spatial_metadata['idx']
        h5_idx = spatial_metadata['file']
        #spatial_feature = self.get_spatial_features_from_h5(h5_file)[spatial_feature_idx]
        filename = f'{self.subset}_gqa_spatial_{h5_idx}.h5'
        h5_file = h5py.File(os.path.join(self.spatial_features_path, filename), 'r')

        spatial_feature = h5_file['features'][spatial_feature_idx]
        #spatial_feature = self.spatial_features[h5_file][spatial_feature_idx]
        return spatial_feature,spatial_metadata

    def get_object_feature(self, imageId,feature_type):
        object_metadata = self.object_info_json[imageId]
        object_feature_idx = object_metadata['idx']
        h5_idx = object_metadata['file']
        #object_feature = self.get_object_features_from_h5(h5_file)[object_feature_idx]
        filename=f'{self.subset}_gqa_objects_{h5_idx}.h5'
        h5_file=h5py.File(os.path.join(self.object_features_path,filename),'r')

        #     os.path(self.object_features_path,,'r') as f:
        #         pass
        object_feature=h5_file[feature_type][object_feature_idx]
        #object_feature = self.object_features[h5_file][feature_type][object_feature_idx]
        return object_feature,object_metadata

