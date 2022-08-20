import warnings

import yaml


class TrainingConfig(object):
    def __init__(self,configs_filename=None, args=None,**kwargs):
        self.AUTO_LR_FIND = False
        self.BATCH_SIZE= None
        self.LR_DECAY_LIST= None
        self.LOSS_REDUCTION= None
        self.LR_BASE= None
        self.MAX_EPOCHS= None
        self.GRAD_NORM_CLIP= None
        self.GRAD_ACCU_STEPS= None
        self.OPT_PARAMS_BETAS= None
        self.OPT_PARAMS_EPS= None
        self.WARMUP_EPOCH= None
        self.LR_DECAY_EPOCH= None
        self.LR_DECAY_R= None
        self.NUM_WORKERS=None
        self.IMAGE_FEATURE_TYPES: None
        self.WEIGHT_DECAY : None
        self.USE_WARMUP: None



        with open(configs_filename) as f:
            configs_yml=yaml.load(f,yaml.FullLoader)
            for key,val in configs_yml.items():
                self._check_attribute(key)
                setattr(self, key, val)

        for key in kwargs:
            self._check_attribute(key)
            setattr(self, key, kwargs[key])

        if args:
            for key,value in vars(args).items():
                self._check_attribute(key)
                setattr(self, key, value)

    def _check_attribute(self, key):
        if not hasattr(self, key):
            warnings.warn(f'{self} has no attribute named {key}', category = UserWarning)
