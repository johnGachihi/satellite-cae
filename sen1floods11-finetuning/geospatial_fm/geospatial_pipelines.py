"""
This file holds pipeline components useful for loading remote sensing images and annotations.
"""
import os.path as osp

import numpy as np
import rioxarray
import torchvision.transforms.functional as F
from mmengine.structures import BaseDataElement as DC, PixelData
from mmseg.structures import SegDataSample
# from mmseg.datasets.builder import PIPELINES
from mmcv.transforms import BaseTransform, TRANSFORMS
from torchvision import transforms
from typing import Sequence, Union
import torch


def open_tiff(fname):
    data = rioxarray.open_rasterio(fname)
    return data.to_numpy()


@TRANSFORMS.register_module()
class ConstantMultiply(BaseTransform):
    """Multiply image by constant.

    It multiplies an image by a constant

    Args:
        constant (float, optional): The constant to multiply by. 1.0 (e.g. no alteration if not specified)
    """

    def __init__(self, constant=1.0):
        self.constant = constant

    def transform(self, results):
        """Call function to multiply by constant input img

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Results with image multiplied by constant
        """

        results["img"] = results["img"] * self.constant

        return results


@TRANSFORMS.register_module()
class BandsExtract(BaseTransform):

    """Extract bands from image. Assumes channels last

    It extracts bands from an image. Assumes channels last.

    Args:
        bands (list, optional): The list of indexes to use for extraction. If not provided nothing will happen.
    """

    def __init__(self, bands=None):
        self.bands = bands

    def transform(self, results):
        """Call function to multiply extract bands

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Results with extracted bands
        """

        if self.bands is not None:
            results["img"] = results["img"][..., self.bands]

        return results


@TRANSFORMS.register_module()
class TorchRandomCrop(BaseTransform):

    """

    It randomly crops a multichannel tensor.

    Args:
        crop_size (tuple): the size to use to crop
    """

    def __init__(self, crop_size=(224, 224)):
        self.crop_size = crop_size

    def transform(self, results):
        i, j, h, w = transforms.RandomCrop.get_params(results["img"], self.crop_size)
        results["img"] = F.crop(results["img"], i, j, h, w).float()
        results["gt_semantic_seg"] = F.crop(results["gt_semantic_seg"], i, j, h, w)

        return results


@TRANSFORMS.register_module()
class TorchNormalize(BaseTransform):
    """Normalize the image.

    It normalises a multichannel image using torch

    Args:
        mean (sequence): Mean values .
        std (sequence): Std values of 3 channels.
    """

    def __init__(self, means, stds):
        self.means = means
        self.stds = stds

    def transform(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = F.normalize(results["img"], self.means, self.stds, False)
        results["img_norm_cfg"] = dict(mean=self.means, std=self.stds)
        return results


@TRANSFORMS.register_module()
class Reshape(BaseTransform):
    """
    It reshapes a tensor.
    Args:
        new_shape (tuple): tuple with new shape
        keys (list): list with keys to apply reshape to
        look_up (dict): dictionary to use to look up dimensions when more than one is to be inferred from the original image, which have to be inputed as -1s in the new_shape argument. eg {'2': 1, '3': 2} would infer the new 3rd and 4th dimensions from the 2nd and 3rd from the original image.
    """

    def __init__(self, new_shape, keys, look_up=None):
        self.new_shape = new_shape
        self.keys = keys
        self.look_up = look_up

    def transform(self, results):
        dim_to_infer = np.where(np.array(self.new_shape) == -1)[0]

        print(f"results gt {results['gt_semantic_seg'].shape}")
        
        for key in self.keys:
            if (len(dim_to_infer) > 1) & (self.look_up is not None):
                old_shape = results[key].shape
                tmp = np.array(self.new_shape)
                for i in range(len(dim_to_infer)):
                    tmp[dim_to_infer[i]] = old_shape[self.look_up[str(dim_to_infer[i])]]
                self.new_shape = tuple(tmp)
            results[key] = results[key].reshape(self.new_shape)

        return results


@TRANSFORMS.register_module()
class CastTensor(BaseTransform):
    """

    It casts a tensor.

    Args:
        new_type (str): torch type
        keys (list): list with keys to apply reshape to
    """

    def __init__(self, new_type, keys):
        self.new_type = new_type
        self.keys = keys

    def transform(self, results):
        for key in self.keys:
            results[key] = results[key].type(self.new_type)

        # print(f"CastTensor: {results}")

        return results


@TRANSFORMS.register_module()
class CollectTestList(BaseTransform):
    """

    It processes the data in a way that conforms with inference and test pipelines.

    Args:

        keys (list): keys to collect (eg img/gt_semantic_seg)
        meta_keys (list): additional meta to collect and add to img_metas

    """

    def __init__(
        self,
        keys,
        meta_keys=(
            "filename",
            "ori_filename",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "img_norm_cfg",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def transform(self, results):
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        img_meta = [img_meta]
        data["img_metas"] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = [results[key]]
        return data

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(keys={self.keys}, meta_keys={self.meta_keys})"
        )


@TRANSFORMS.register_module()
class TorchPermute(BaseTransform):
    """Permute dimensions.

    Particularly useful in going from channels_last to channels_first

    Args:
        keys (Sequence[str]): Keys of results to be permuted.
        order (Sequence[int]): New order of dimensions.
    """

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def transform(self, results):
        for key in self.keys:
            results[key] = results[key].permute(self.order)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(keys={self.keys}, order={self.order})"


@TRANSFORMS.register_module()
class LoadGeospatialImageFromFile(BaseTransform):
    """

    It loads a tiff image. Returns in channels last format.

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        nodata (float/int): no data value to substitute to nodata_replace
        nodata_replace (float/int): value to use to replace no data
    """

    def __init__(self, to_float32=False, nodata=None, nodata_replace=0.0):
        self.to_float32 = to_float32
        self.nodata = nodata
        self.nodata_replace = nodata_replace

    def transform(self, results):
        # print(f"LoadGeospatialImageFromFile: {results}")
        # if results.get("img_prefix") is not None:
        #     filename = osp.join(results["img_prefix"], results["img_info"]["filename"])
        # else:
        #     filename = results["img_info"]["filename"]
        img = open_tiff(results["img_path"])
        # to channels last format
        img = np.transpose(img, (1, 2, 0))

        if self.to_float32:
            img = img.astype(np.float32)

        if self.nodata is not None:
            img = np.where(img == self.nodata, self.nodata_replace, img)

        results["filename"] = results["img_path"]  # filename
        results["ori_filename"] = results["img_path"] # results["img_info"]["filename"]
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        results["flip"] = False
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}"
        return repr_str


@TRANSFORMS.register_module()
class LoadGeospatialAnnotations(BaseTransform):
    """Load annotations for semantic segmentation.

    Args:
        to_uint8 (bool): Whether to convert the loaded label to a uint8
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        nodata (float/int): no data value to substitute to nodata_replace
        nodata_replace (float/int): value to use to replace no data


    """

    def __init__(
        self,
        reduce_zero_label=False,
        nodata=None,
        nodata_replace=-1,
    ):
        self.reduce_zero_label = reduce_zero_label
        self.nodata = nodata
        self.nodata_replace = nodata_replace

    def transform(self, results):
        # print(f"LoadGeospatialAnnotations: {results}")

        # if results.get("seg_prefix", None) is not None:
        #     filename = osp.join(results["seg_prefix"], results["ann_info"]["seg_map"])
        # else:
        #     filename = results["ann_info"]["seg_map"]

        filename = results["seg_map_path"]

        gt_semantic_seg = open_tiff(filename).squeeze()

        if self.nodata is not None:
            gt_semantic_seg = np.where(
                gt_semantic_seg == self.nodata, self.nodata_replace, gt_semantic_seg
            )
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        if results.get("label_map", None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results["label_map"].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id

        results["gt_semantic_seg"] = gt_semantic_seg
        results["seg_fields"].append("gt_semantic_seg")
        return results


@TRANSFORMS.register_module()
class PackSegInputs_(BaseTransform):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        print(f"PackSegData results {results['ori_shape']}")
        packed_results = dict()

        img = results["img"]
        packed_results["inputs"] = img

        data_sample = SegDataSample()        
        gt_sem_seg_data = dict(data=results["gt_semantic_seg"])
        data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)
        packed_results["data_samples"] = data_sample

        # print(f"inputs shape: {packed_results['inputs'].shape}")
        # print(f"PackSegInput packed_result: {packed_results}")

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int,
                float]) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: the converted data.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data.copy())
    elif isinstance(data, Sequence) and not mmengine.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')
    

@TRANSFORMS.register_module()
class ToTensor_(BaseTransform):
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Required keys:

    - all these keys in `keys`

    Modified Keys:

    - all these keys in `keys`

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """

    def __init__(self, keys: Sequence[str]) -> None:
        self.keys = keys

    def transform(self, results: dict) -> dict:
        """Transform function to convert data to `torch.Tensor`.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: `keys` in results will be updated.
        """
        for key in self.keys:

            key_list = key.split('.')
            cur_item = results
            for i in range(len(key_list)):
                if key_list[i] not in cur_item:
                    raise KeyError(f'Can not find key {key}')
                if i == len(key_list) - 1:
                    cur_item[key_list[i]] = to_tensor(cur_item[key_list[i]])
                    break
                cur_item = cur_item[key_list[i]]

        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(keys={self.keys})'
