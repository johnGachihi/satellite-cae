import os
import torch

from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from furnace.transforms import RandomResizedCropAndInterpolationWithTwoPic
from timm.data import create_transform

from dall_e.utils import map_pixels
from furnace.masking_generator import MaskingGenerator, RandomMaskingGenerator
from furnace.dataset_folder import ImageFolder, SentinelIndividualImageDataset, SentinelNormalize


def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x

class DataAugmentationForCAE(object):
    def __init__(self, args):
        self.mean = (1370.19151926, 1184.3824625 , 1120.77120066, 1136.26026392,
                 1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
                 1972.62420416,  582.72633433,   14.77112979, 1732.16362238, 1247.91870117)
        self.std = (633.15169573,  650.2842772 ,  712.12507725,  965.23119807,
                948.9819932 , 1108.06650639, 1258.36394548, 1233.1492281 ,
                1364.38688993,  472.37967789,   14.3114637 , 1310.36996126, 1087.6020813)

        self.args = args
        
        if args.mask_generator == 'block':
            self.masked_position_generator = MaskingGenerator(
                args.window_size, num_masking_patches=args.num_mask_patches,
                max_num_patches=args.max_mask_patches_per_block,
                min_num_patches=args.min_mask_patches_per_block,
            )
        elif args.mask_generator == 'random':
            self.masked_position_generator = RandomMaskingGenerator(
                args.window_size, ratio_masking_patches=args.ratio_mask_patches
            )
        

    def __call__(self, image):
        image = SentinelNormalize(self.mean, self.std)(image)
        image = transforms.ToTensor()(image)
        image = transforms.RandomHorizontalFlip()(image)
        image = transforms.RandomResizedCrop(
            self.args.input_size,
            scale=(0.2, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC
        )(image)
        
        return image, self.masked_position_generator()
    

    def __repr__(self):
        repr = "(DataAugmentationForCAE,\n"
        # TODO
        # repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_cae_pretraining_dataset(args):
    transform = DataAugmentationForCAE(args)
    print("Data Aug = %s" % str(transform))
    return ImageFolder(args.data_path, transform=transform)

def build_cae_pretraining_satellite_dataset(is_train, args):
    transform = DataAugmentationForCAE(args)
    print("Data Aug = %s" % str(transform))
    dataset = SentinelIndividualImageDataset(args.data_path, args.csv_path, transform, masked_bands=args.masked_bands,
                                             dropped_bands=args.dropped_bands)
    return dataset

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
