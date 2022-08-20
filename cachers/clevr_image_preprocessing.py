import glob
import os
from skimage import io
from PIL import Image, ImageOps
import numpy as np
from skimage.color import rgba2rgb
from tqdm import tqdm
import utils
import torchvision.transforms.functional as F


def cache_features(images_dir, cache_dir, subset):
    if not os.path.exists(os.path.join(cache_dir, subset)):
        os.makedirs(os.path.join(cache_dir, subset))

    # with open(os.path.join(questions_dir, f'CLEVR_{subset}_questions.json'), 'r') as json_file:
    #     questions = json.load(json_file)['questions']
    for image_filename in tqdm(glob.glob(os.path.join(images_dir, subset, '*.png'))):
    #for v in tqdm(questions_subset):
        # image = rgba2rgb(io.imread(image_filename)).astype(float)  # / 255
        # image = Image.open(image_filename)
        name, ext = os.path.splitext(os.path.basename(image_filename))

        image = rgba2rgb(io.imread(image_filename)).astype(float)  # / 255
        # image = ImageOps.pad(image,[0,80])
        # image = image.resize([128,128],Image.BILINEAR)
        image = F.to_tensor(image)
        image = F.pad(image, [0, 80])
        image = F.resize(image, [128, 128])
        image = F.normalize(image, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.224])
        np.save(os.path.join(cache_dir, subset, name + '.npy'), image)


if __name__ == '__main__':
    questions_dir = '/mnt/ssd2/Datasets/CLEVR/CLEVR_v1.0/questions'
    # images_dir = '/media/hybridsyntax/Workspace/Datasets/CLEVR/resnet101_features'
    cache_dir = '/mnt/ssd2/Datasets/CLEVR/preprocessed'
    images_dir = '/mnt/ssd2/Datasets/CLEVR/CLEVR_v1.0/images'

    # questions_dir = '/mnt/ssd2/Datasets/CLEVR/CLEVR_v1.0/questions'

    # images.append(image)

    # cache_dir = '/mnt/ssd2/Datasets/CLEVR/resnet101_h5'
    # # vocabs_dir= '/home/hybridsyntax/Datasets/cache/vocabs'
    # if not os.path.exists(cache_dir):
    #     os.makedirs(cache_dir)

    # cache_features(images_dir, cache_dir, subset = 'val')
    # cache_features(questions_dir, images_dir, cache_dir, subset = 'test')
    cache_features(images_dir, cache_dir, subset = 'val')
    cache_features(images_dir, cache_dir, subset = 'train')
    cache_features(images_dir, cache_dir, subset = 'test')
