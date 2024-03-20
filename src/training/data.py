import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value
import blosc
from functools import partial

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

from open_clip.tokenizer import decode
from .instruct.codes import (
    CodeLabelPredictionTask,
    SingleCodeLabelPredictionTask,
    CodeLabelPredictionTaskEvaluation
)
from .instruct.codes.descriptions import ICDDescription, PHEDescription
from .instruct.codes.processing import (
    EncounterDataframeProcess,
    NegativeCodeCacheSampling,
    GroupBySampling,
    ICDConvert,
    PHEConvert,
)
from .instruct.utils import (
    get_code_label_prediction_instruction_template,
)
from .instruct.codes.processing.data_bins import AgglomerativeDataBins, MonthDataBins
from .instruct import (
    InstructTasks,
    GPT2InstructTokenizer,
    InstructTokenizer
)

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist), \
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
            self,
            urls,
            weights=None,
            nshards=sys.maxsize,
            worker_seed=None,
            deterministic=False,
            epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights), \
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    pipeline_extension = [
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ]

    return get_wds_data_info(
        args=args,
        pipeline_extension=pipeline_extension,
        is_train=is_train,
        epoch=epoch,
        floor=floor
    )


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_instruct_tokenizer(model_name, tokenizer, pad_id, max_seq_length, ignore_index):
    if 'gpt' in model_name:
        return GPT2InstructTokenizer(
            tokenizer=tokenizer, pad_id=pad_id, max_seq_length=max_seq_length, ignore_index=ignore_index
        )
    else:
        return InstructTokenizer(
            tokenizer=tokenizer, pad_id=pad_id, max_seq_length=max_seq_length, ignore_index=ignore_index
        )


def get_code_prediction_instruction_template():
    return get_code_label_prediction_instruction_template()

def get_code_label_prediction_task(
        training_type,
        encounter_dataframe_process,
        negative_code_sampling,
        dataframe_sampling,
        code_label_prediction_instructions,
        time_bins,
        code_convert,
        patient_id_column,
        code_column,
        position_column
):

    if training_type == 'single':
        return SingleCodeLabelPredictionTask(
            encounter_dataframe_process=encounter_dataframe_process,
            dataframe_sampling=dataframe_sampling,
            code_instructions=code_label_prediction_instructions,
            time_bins=time_bins,
            code_convert=code_convert,
            negative_code_sampling=negative_code_sampling,
            patient_id_column=patient_id_column,
            code_column=code_column,
            position_column=position_column,
        )
    elif training_type == 'all':
        return CodeLabelPredictionTask(
            encounter_dataframe_process=encounter_dataframe_process,
            dataframe_sampling=dataframe_sampling,
            code_instructions=code_label_prediction_instructions,
            time_bins=time_bins,
            code_convert=code_convert,
            negative_code_sampling=negative_code_sampling,
            patient_id_column=patient_id_column,
            code_column=code_column,
            position_column=position_column,
        )
    elif training_type == 'trajectory':
        raise NotImplementedError()
    else:
        raise ValueError('Invalid training tpe')

def get_code_label_prediction_task_eval(
        encounter_dataframe_process,
        negative_code_sampling,
        dataframe_sampling,
        code_label_prediction_instructions,
        time_bins,
        code_convert,
        patient_id_column,
        code_column,
        position_column
):

    return CodeLabelPredictionTaskEvaluation(
        encounter_dataframe_process=encounter_dataframe_process,
        dataframe_sampling=dataframe_sampling,
        code_instructions=code_label_prediction_instructions,
        time_bins=time_bins,
        code_convert=code_convert,
        negative_code_sampling=negative_code_sampling,
        patient_id_column=patient_id_column,
        code_column=code_column,
        position_column=position_column,
    )


def get_code_convert(args):
    if 'phe' in args.code_column:
        return PHEConvert(
            descriptions=PHEDescription(),
            lowercase=False
        )
    else:
        return ICDConvert(
            descriptions=ICDDescription(),
            billable_probability=args.billable_probability,
            top_non_probability=args.top_non_probability,
            mixed_non_probability=args.mixed_non_probability,
            lowercase=False
        )

def get_encounter_dataframe(encounter_file):
    return pd.read_parquet(
        encounter_file, columns=['PatientID', 'ContactDTS', 'ICD10CD', 'phecode']
    )

def get_wds_dataset_icd_instruct(
        args,
        preprocess_img,
        is_train,
        epoch=0,
        floor=False,
        tokenizer=None,
        return_sample=False,
        eval_mode=False,
        encounter_dataframe=None
):

    if encounter_dataframe is None:
        encounter_dataframe = get_encounter_dataframe(encounter_file=args.encounter_file)

    dataframe_sampling = GroupBySampling()

    code_convert = get_code_convert(args=args)

    encounter_dataframe_process = EncounterDataframeProcess(
        encounter_dataframe=encounter_dataframe,
        patient_id_column=args.patient_id_column,
        contact_date_column=args.contact_date_column,
        time_difference_column=args.time_difference_column,
        code_column=args.code_column,
        position_column=args.position_column,
    )

    # Set these to - number_of_instructions * k_shot?
    negative_code_cache_sampling = NegativeCodeCacheSampling(
        encounter_dataframe_process=encounter_dataframe_process,
        negatives_type=args.negatives_type,
        code_task_negative_cache_size=100,
        minimum_negatives_size=10,
    )

    time_bins = [
        AgglomerativeDataBins(distance_threshold=distance_threshold)
        for distance_threshold in args.distance_threshold
    ]

    code_label_prediction_instructions = get_code_prediction_instruction_template()

    if args.eval_mode or eval_mode:
        code_label_prediction_task = get_code_label_prediction_task_eval(
            encounter_dataframe_process=encounter_dataframe_process,
            negative_code_sampling=negative_code_cache_sampling,
            dataframe_sampling=dataframe_sampling,
            code_label_prediction_instructions=code_label_prediction_instructions,
            time_bins=time_bins,
            code_convert=code_convert,
            patient_id_column=args.patient_id_column,
            code_column=args.code_column,
            position_column=args.position_column
        )
    else:
        code_label_prediction_task = get_code_label_prediction_task(
            training_type=args.training_type,
            encounter_dataframe_process=encounter_dataframe_process,
            negative_code_sampling=negative_code_cache_sampling,
            dataframe_sampling=dataframe_sampling,
            code_label_prediction_instructions=code_label_prediction_instructions,
            time_bins=time_bins,
            code_convert=code_convert,
            patient_id_column=args.patient_id_column,
            code_column=args.code_column,
            position_column=args.position_column
        )

    if tokenizer is None:
        instruct_tokenizer = None
    else:
        instruct_tokenizer = get_instruct_tokenizer(
            tokenizer=tokenizer,
            model_name=args.model,
            pad_id=args.pad_id,
            max_seq_length=args.max_seq_length,
            ignore_index=-100
        )

    instruct_tasks = InstructTasks(
        code_instruct_task=code_label_prediction_task,
        instruct_tokenizer=instruct_tokenizer
    )

    instruct_function = partial(
        instruct_tasks.process_sample_from_args, args=args
    )

    image_key, text_key = get_sample_keys(args)
    if return_sample or args.eval_mode:
        rename = wds.rename(image=image_key, text=text_key, labels=text_key)
        return_tuple = wds.to_tuple("image", "text", "labels")
    else:
        rename = wds.rename(image=image_key, text=text_key)
        return_tuple = wds.to_tuple("image", "text")

    torch_blosc_convert = get_torch_blosc_convert(args)

    # Build pipeline extension
    pipeline_extension = [
        wds.decode(
            wds.handle_extension("blosc", blosc.unpack_array),
        ),
        rename,
        wds.map_dict(text=instruct_function),
        wds.map_dict(image=torch_blosc_convert),
        wds.map_dict(image=preprocess_img),
        return_tuple,
        wds.batched(args.batch_size, partial=not is_train),
    ]

    return get_wds_data_info(
        args=args,
        pipeline_extension=pipeline_extension,
        is_train=is_train,
        epoch=epoch,
        floor=floor
    )

def get_sample_keys(args):
    if 'scatter' in args.model.lower():
        return 'dict.x.blosc', 'dict.meta.pyd'
    elif 'cyto' in args.model.lower():
        return 'x.blosc', 'meta.pyd'
    else:
        raise ValueError('args.model should contain either ecg or cyto')
def get_torch_blosc_convert(args):
    if 'scatter' in args.model.lower():
        return wds_ecg_to_torch
    elif 'cyto' in args.model.lower():
        return wds_cytometry_to_torch
    else:
        raise ValueError('args.name should contain either ecg or cyto')

def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "icddataset":
        return get_wds_dataset_icd_instruct
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data

# def get_icd_instruct_validation_data(args, preprocess_fns, epoch=0, tokenizer=None, do_train=True):
#     preprocess_train, preprocess_val = preprocess_fns
#     data = {}
#
#     if args.val_data:
#         if args.dataset_type == "icddataset":
#             data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
#                 args, preprocess_val, is_train=False, tokenizer=tokenizer)
#         else:
#             data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
#                 args, preprocess_val, is_train=False, tokenizer=tokenizer)
#
#     return data



def wds_ecg_to_torch(x):
    # time in last dimension, shape (12, 2500)
    return torch.from_numpy(x.transpose(1, 0))

def wds_cytometry_to_torch(x):
    # this uses only the first three dimensions:
    # return torch.from_numpy(x[:3, ].astype(np.float32))
    # this uses all dimensions in the data
    return torch.from_numpy(x.astype(np.float32))


def get_number_of_shards_samples(input_shards, is_train, train_num_samples, val_num_samples):
    num_shards = None
    if is_train:
        if train_num_samples is not None:
            num_samples = train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = val_num_samples or 0
    return num_shards, num_samples


def get_initial_pipeline(args, input_shards, resampled, is_train, shared_epoch):
    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with " \
                          "--dataset-resampled)."

    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])

    return pipeline


def get_global_batch_size(args):
    return args.batch_size * args.world_size


def get_number_of_workers(args):
    return max(1, args.workers)


def get_dataset(pipeline, is_train, num_worker_batches=None):
    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    return dataset


def get_wds_dataloader(dataset, num_workers, num_batches, num_samples):
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return dataloader


def get_number_of_worker_batches(
        args,
        input_shards,
        resampled,
        global_batch_size,
        num_workers,
        num_shards,
        num_samples,
        floor
):
    if not resampled:
        num_shards = num_shards or len(expand_urls(input_shards)[0])
        assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    num_batches = round_fn(num_samples / global_batch_size)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    return num_worker_batches


def get_wds_data_info(args, pipeline_extension, is_train, epoch=0, floor=False):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None

    resampled = getattr(args, 'dataset_resampled', False) and is_train

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    num_shards, num_samples = get_number_of_shards_samples(
        input_shards=input_shards,
        is_train=is_train,
        train_num_samples=args.train_num_samples,
        val_num_samples=args.val_num_samples
    )

    pipeline = get_initial_pipeline(
        args=args,
        input_shards=input_shards,
        resampled=resampled,
        is_train=is_train,
        shared_epoch=shared_epoch
    )

    pipeline.extend(pipeline_extension)

    if is_train:
        global_batch_size = get_global_batch_size(args)
        num_workers = get_number_of_workers(args)
        num_worker_batches = get_number_of_worker_batches(
            args=args,
            input_shards=input_shards,
            resampled=resampled,
            global_batch_size=global_batch_size,
            num_shards=num_shards,
            num_samples=num_samples,
            num_workers=num_workers,
            floor=floor
        )
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
    else:
        num_worker_batches = None
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataset = get_dataset(pipeline=pipeline, is_train=is_train, num_worker_batches=num_worker_batches)

    dataloader = get_wds_dataloader(
        dataset=dataset,
        num_workers=args.workers,
        num_batches=num_batches,
        num_samples=num_samples
    )

    print(f'#### constructed dataloader (batches, samples): ####')
    print(f'#### {num_batches}, {num_samples} ####')

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


