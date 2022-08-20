import warnings

import yaml


class ModelConfig(object):
    def __init__(self, FRCN_FEAT_SIZE=None, GRID_FEAT_SIZE=None, BBOX_FEAT_SIZE=None, configs_filename=None, **kwargs):
        #params
        self.FRCN_FEAT_SIZE= FRCN_FEAT_SIZE#(100, 2048)
        self.GRID_FEAT_SIZE = GRID_FEAT_SIZE#(196, 1024)
        self.BBOX_FEAT_SIZE=BBOX_FEAT_SIZE#= (100, 5)



        self.IMG_FEAT_SIZE = 2048
        self.HIDDEN_SIZE = 512
        self.DROPOUT_R = 0.2
        self.CLASSIFER_DROPOUT_R = 0.5
        self.FLAT_OUT_SIZE = 1024 #FLAT_OUT_SIZE: 2048 in yml file

#        self.USE_BBOX_FEAT=True
#        self.BBOXFEAT_EMB_SIZE= 1024


        #in cfg yml file
        self.HIDDEN_SIZE= 1024
        self.FLAT_OUT_SIZE= 2048
        self.USE_BBOX_FEAT= True
        self.BBOXFEAT_EMB_SIZE= 1024
        self.IMG_FEAT_SIZE= 1024

        #execution
        #self.LOSS_REDUCTION= 'sum'
        #self.GRAD_NORM_CLIP= 0.25


        #inherit
        self.WORD_EMBED_SIZE = 300
        self.USE_GLOVE = True

        with open(configs_filename) as f:
            configs_yml=yaml.load(f,yaml.FullLoader)
            for key,val in configs_yml.items():
                self._check_attribute(key)
                setattr(self, key, val)

        for key in kwargs:
            self._check_attribute(key)
            setattr(self, key, kwargs[key])

    def _check_attribute(self, key):
        if not hasattr(self, key):
            warnings.warn(f'{self} has no attribut named {key}', category = RuntimeWarning)
