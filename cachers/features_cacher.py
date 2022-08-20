import argparse
import os
import pickle
import subprocess
from collections import defaultdict
from glob import glob
import json
import h5py
from tqdm import tqdm
import utils


def cache_features(dataset_dir, subset, info_filename):

    spatials_files = defaultdict(list)
    objects_files = defaultdict(list)

    #metadata_json = 'cache/cached_image_features_balanced_metadata.json'
    question_json_filename = os.path.join(
        dataset_dir, f'questions1.2/{subset}_questions.json')
    # os.path.join(dataset_dir, 'objects/gqa_objects_info.json')
    objects_json_filename = info_filename

    # spatial_loc=r'E:\Datasets\GQA\spatial\*.h5'
    # object_loc = r'E:\Datasets\GQA\objects\*.h5'
    # spatial_h5_files = glob(spatial_loc)
    # object_h5_files =glob(object_loc)

    with open(question_json_filename) as json_file:
        question_json = json.load(json_file)
        imageIds = set([v['imageId'] for k, v in question_json.items()])
        #assert len(imageIds)==len(question_json)

    with open(objects_json_filename) as json_file:
        objects = json.load(json_file)

    print(f'There are {len(imageIds)} unique image ids')
    print(f'There are  {len(objects)} features')
    print(
        f'After caching  there will be {len(objects)//len(imageIds)} times less features!')

    for imageId in imageIds:
        _object_file = objects[imageId]['file']
        objects_files[_object_file].append((imageId, objects[imageId]))

    print(objects_files.keys())
    for k, v in objects_files.items():
        print(f'File {k} has {len(v)} entries')

    return objects_files


def merge_jsons(json_dir, original_json_filename, subset):
    merged_dict = {}
    with open(original_json_filename) as original_json_file:
        original_json = json.load(original_json_file)

    # merged_name=f'{subset}_{os.path.basename(original_json_filename).split(".")[0]}.json'
    merged_name = f'{subset}_{os.path.basename(original_json_filename)}'
    for json_filename in glob(os.path.join(json_dir, '*.json')):
        with open(json_filename) as json_file:
            for k, v in json.load(json_file).items():
                merged_dict[k] = v
                merged_dict[k]['original_idx'] = original_json[k]['idx']
    with open(os.path.join(json_dir, merged_name), 'w') as merged_json_file:
        json.dump(merged_dict, merged_json_file)
    print('All merged into ', os.path.join(json_dir, merged_name))


def extract_hf5(hf5_filenme, selected_features, subset, cache_dir, spatial_features=False):
    cache_dir = os.path.join(cache_dir, subset)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    image2idx = {}
    object_feats = []
    bboxes_feats = []
    feat_count = 0

    hf5_cache_filenme = f'{cache_dir}/{subset}_' + \
        os.path.split(hf5_filenme)[-1]

    if os.path.exists(hf5_cache_filenme):
        print(hf5_cache_filenme + ' exists')
        with h5py.File(hf5_cache_filenme, 'r') as cached_h5f:
            assert len(cached_h5f['features']) == len(cached_h5f['bboxes'])
            length = len(cached_h5f['features'])
        return length

    #print(f'Processing {hf5_filenme}', )
    # hf5_filenme=hf5_filenme.replace('ds/','')
    with h5py.File(hf5_filenme, 'r') as hf5file:
        assert len(selected_features) <= len(hf5file['features'])
        if not spatial_features:
            assert len(selected_features) <= len(hf5file['bboxes'])
        for i, data in enumerate(tqdm(selected_features, desc=f'Processing {hf5_filenme}')):
            _imageId, metadata = data
            orig_idx = metadata['idx']
            try:
                object_feats.append(hf5file['features'][orig_idx])
                if not spatial_features:
                    bboxes_feats.append(hf5file['bboxes'][orig_idx])
            except IndexError:
                print('\ncurrent i:', i)
                print('imageId: ', _imageId)
                print('info: ', metadata)
                input('Press to continue')
            image2idx[_imageId] = metadata
            image2idx[_imageId]['original_idx'] = orig_idx
            image2idx[_imageId]['idx'] = i
            feat_count = feat_count + 1

        #utils.save_json(image2idx, f'cache/objects/{subset}_{hf5_filenme}.json')
        utils.save_json(image2idx, hf5_cache_filenme.split('.')[0]+'.json')

        with h5py.File(hf5_cache_filenme, 'w') as cached_h5f:
            cached_h5f.create_dataset(
                'features', data=object_feats, compression='lzf')
            if not spatial_features:
                cached_h5f.create_dataset(
                    'bboxes', data=bboxes_feats, compression='lzf')
    return feat_count


def unzip_objects(hf5_filenme):
    if os.path.exists('objectFeatures.zip'):
        if not os.path.exists('objects'):
            os.makedirs('objects')
        hf5_filenme_zz = hf5_filenme.replace('ds/', '')
        subprocess.Popen(['unzip', 'objectFeatures.zip', hf5_filenme_zz])

        # with zipfile.ZipFile('objectFeatures.zip') as z:
        # with open(hf5_filenme, 'wb') as f:
        # f.write(z.read(hf5_filenme_zz))


def cache_object_features(object_files, subset, dataset_loc):
    for hf5_idx, image_features in object_files.items():
        print(f'Hf5 Index {hf5_idx}')
        hf5_filenme = os.path.join(
            dataset_loc, f'objects/gqa_objects_{str(hf5_idx)}.h5')
        unzip_objects(hf5_filenme)
        feat_count = extract_hf5(hf5_filenme, image_features, subset)
        if os.path.exists('objectFeatures.zip'):
            # os.remove(hf5_filenme)
            subprocess.Popen(['rm', hf5_filenme])
        #assert feat_count==len(image_features)


# def cache_spatial_features(file_info, spatial_h5_files, spatials_files,subset):
#     print(spatials_files.keys())
#     image2idx = {}
#     spat_feats = []
#     feat_count = 0
#     for hf5_idx, image_features in spatials_files.items():
#         #hf5_filenme = filter(spatial_h5_files, f'gqa_spatial_{str(hf5_idx)}.h5')[-1]
#
#         hf5_filenme= spatial_h5_files[spatial_h5_files.index(f'E:\Datasets\GQA\spatial\gqa_spatial_{str(hf5_idx)}.h5')]
#         print(f'Processing {hf5_filenme}', )
#         hf5_cache_filenme = f'cache/spatial/{subset}_' + os.path.split(hf5_filenme)[-1]
#         if  os.path.exists(hf5_cache_filenme):
#             print(hf5_cache_filenme+' exists')
#             continue
#         with h5py.File(hf5_filenme, 'r') as hf5file:
#             for i, data in enumerate(tqdm(image_features)):
#                 imageId, metadata = data
#                 orig_idx = metadata['spatial']['idx']
#                 spat_feats.append(hf5file['features'][orig_idx])
#                 image2idx[imageId] = metadata
#                 image2idx[imageId]['idx'] = i
#                 feat_count = feat_count + 1
#
#             utils.save_json(image2idx, f'cache/spatial/spatial_info_{subset}_{hf5_idx}.json')
#             with h5py.File(hf5_cache_filenme, 'w') as cachedh5f:
#                 cachedh5f.create_dataset('features', data = spat_feats, compression = 'lzf')
#     assert feat_count == len(file_info)
#
def merge_jzons(subset):
    jsonz = f'objects/{subset}'
    refrence_json = os.path.join(dataset_dir, 'objects/gqa_objects_info.json')
    merge_jsons(jsonz, refrence_json, subset)


def validate_cache(json_filename, objects_dir, cache_dir, subset, feature_type):
    cache_dir = os.path.join(cache_dir, subset)
    json_filename = os.path.join(cache_dir, json_filename)
    logs = ""
    final = True
    with open(json_filename) as json_file:
        objects_info = json.load(json_file)
    for k, v in tqdm(objects_info.items(), 'Validating cache'):
        h5idx = v['file']
        idx = v['idx']
        original_idx = v['original_idx']
        original_h5_filename = os.path.join(
            objects_dir, f"gqa_{feature_type}_{h5idx}.h5")
        cached_h5_filename = os.path.join(
            cache_dir, f"{subset}_gqa_{feature_type}_{h5idx}.h5")
        assert os.path.exists(original_h5_filename) == os.path.exists(
            cached_h5_filename)
        with h5py.File(cached_h5_filename) as cached_file:
            with h5py.File(original_h5_filename) as original_file:
                result = (cached_file['features'][idx] ==
                          original_file['features'][original_idx]).all()
                final = final & result
                log = f'{k},{idx},{os.path.basename(cached_h5_filename)},{original_idx},{os.path.basename(original_h5_filename)},{result}\n'
                logs = logs+log
    print(f'{subset} features are ', result)
    with open(os.path.join(cache_dir, 'logs.csv'), 'w') as log_file:
        log_file.write(logs)


def trim_questions(questions_dir, cache_dir):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    for json_filename in glob(os.path.join(questions_dir, '*.json')):
        trimmed_questions = {}
        with open(json_filename, 'r') as json_file:
            questions = json.load(json_file)
            for k, v in tqdm(questions.items(), 'Processing '+os.path.basename(json_filename)):
                trimmed_questions[k] = {'question': v.get('question'),
                                        'answer': v.get('answer'),
                                        'fullAnswer': v.get('fullAnswer'),
                                        'imageId': v.get('imageId')}

        with open(os.path.join(cache_dir, 'trimmed_'+os.path.basename(json_filename)), 'w') as json_file:
            json.dump(trimmed_questions, json_file)


if __name__ == '__main__':
    # trim_questions(r'E:\Datasets\GQA\questions1.2\balanced','cache/questions1.2')
    # exit()

    parser = argparse.ArgumentParser(description='Personal information')
    parser.add_argument('--dataset_dir', type=str,)
    parser.add_argument('--cache_dir', type=str, )
    parser.add_argument('--subset', type=str)
    parser.add_argument('--action', type=str)
    parser.add_argument('--feature_type', type=str)
    parser.add_argument('--hf_idx', type=int)
    parser.add_argument('--info_filename', type=str)
    # parser.add_argument('--merge_jsons', action = argparse.BooleanOptionalAction)
    # parser.add_argument('--validate', action = argparse.BooleanOptionalAction)

    args = parser.parse_args()
    if args.action == 'cache':
        objects_files = cache_features(
            args.dataset_dir, args.subset, args.info_filename)
        with open(os.path.join(args.cache_dir, f'{args.subset}_spatial_files.pkl'), 'wb') as f:
            pickle.dump(objects_files, f)
    elif args.action == 'extract':
        with open(os.path.join(args.cache_dir, f'{args.subset}_spatial_files.pkl'), 'rb') as f:
            objects_files = pickle.load(f)
            for hf5_idx, image_features in objects_files.items():
                print(f'Hf5 Index {hf5_idx}')
                extract_hf5(os.path.join(args.dataset_dir, f'{args.feature_type}/gqa_{args.feature_type}_{hf5_idx}.h5'), objects_files[hf5_idx],
                            args.subset, args.cache_dir,
                            args.feature_type == 'spatial')
    elif args.action == 'merge_jsons':
        merge_jsons(os.path.join(args.cache_dir, args.subset),
                    os.path.join(args.dataset_dir, args.feature_type,
                                 'gqa_spatial_info.json'),
                    args.subset)
    elif args.action == 'validate':
        validate_cache(f'{args.subset}_gqa_{args.feature_type}_info.json', os.path.join(args.dataset_dir, args.feature_type),
                       f'cache/{args.feature_type}',
                       args.subset, args.feature_type)

    # python
    # features_cacher.py - -dataset_dir
    # E:\Datasets\GQA\ --subset
    # train_balanced - -cache_dir = cache / spatial - -info_filename = E:\Datasets\GQA\spatial\gqa_spatial_info.json -
    # -action = validate - -feature_type = spatial

    #dataset_dir = r'E:\Datasets\GQA'
    # if args.action=='cache':
    #     cache_features(args.dataset_dir,args.subset)
    # elif args.action=='merge':
    #     merge_jzons(args.subset)
    # elif args.action=='validate':
    #     validate_cache(f'{args.subset}_gqa_objects_info.json',
    #                    os.path.join(args.dataset_dir,'objects'),r'cache\objects',args.subset)
    # validate_cache(r'testdev_balanced_gqa_objects_info.json', os.path.join(dataset_dir, 'objects'), r'cache\objects',
    #                'testdev_balanced')
    # validate_cache(r'train_balanced_gqa_objects_info.json', os.path.join(dataset_dir, 'objects'), r'cache\objects',
    #                'train_balanced')
