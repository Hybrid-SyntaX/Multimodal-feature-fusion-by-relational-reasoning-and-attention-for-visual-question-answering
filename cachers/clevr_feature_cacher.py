import json
import os
import shutil

import h5py
import numpy as np
from tqdm import tqdm


def cache_features(questions_dir, images_dir, cache_dir, subset):
    # if not os.path.exists(os.path.join(cache_dir, subset)):
    #     os.makedirs(os.path.join(cache_dir, subset))
    with open(os.path.join(questions_dir, f'CLEVR_{subset}_questions.json'), 'r') as json_file:
        questions = json.load(json_file)['questions']
        # chunk_size = len(questions) // 5
        # chunk_remains = len(questions) % 5
        # assert  (chunk_size*5 + chunk_remains) == len(questions)
        #chunks = [len(lst) for lst in np.array_split(np.arange(0, len(questions)), 5)]
        with h5py.File(os.path.join(cache_dir, subset) + '.h5', 'w') as h5file:
            h5_dataset=h5file.create_dataset('features',(len(questions),196,1024), #196,1024
                                             maxshape=(None,196,1024),
                                             compression = 'lzf',
                                             dtype = float,
                                             )


            features = []

            # h5_len=0
            # for i,v in enumerate(tqdm(questions)):
            h5_len = 0
            idx=0
            img2h5idx=dict()
            for questions_subset in tqdm(np.array_split(questions, 20)):
                features = []
                for v in tqdm(questions_subset):
                    feature_file = os.path.join(images_dir, subset, str(v['image_index'])) + '.npz'
                    # shutil.copy(feature_file,os.path.join(cache_dir,subset))

                    feature = np.load(feature_file)
                    features.append(feature['x'])
                    img2h5idx[v['image_index']]=idx
                    idx=idx+1
                #h5_dataset[i]=feature['x']

                #if len(features)== chunk_size or len(questions)-i-1 ==chunk_remains:
                h5_dataset[h5_len:h5_len + len(features)] = features
                h5_len = h5_len + len(features)
                print(h5_len)
                features.clear()
            print(h5_len,len(questions))
            assert h5_len==len(questions)
            with open(os.path.join(cache_dir, subset) + '.json', 'w') as json_map_file:
                json.dump(img2h5idx,json_map_file)
            print(f'Caching {subset} was successful')







if __name__ == '__main__':
    # questions_dir = '/media/hybridsyntax/Workspace/Datasets/CLEVR/CLEVR-Humans'
    # images_dir = '/media/hybridsyntax/Workspace/Datasets/CLEVR/resnet101_features'
    # cache_dir = '../cache/clevr_humans_resnet101'

    questions_dir = '/mnt/ssd2/Datasets/CLEVR/CLEVR_v1.0/questions'
    images_dir = '/mnt/shared/clevr/resnet101'
    cache_dir = '/mnt/ssd2/Datasets/CLEVR/resnet101_h5'
    # vocabs_dir= '/home/hybridsyntax/Datasets/cache/vocabs'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_features(questions_dir, images_dir, cache_dir, subset = 'val')
    #cache_features(questions_dir, images_dir, cache_dir, subset = 'test')
    cache_features(questions_dir, images_dir, cache_dir, subset = 'train')
