# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import hashlib
import math
import json
from pathlib import Path
import os

import julius
import torch as th
from torch import distributed
import torchaudio as ta
from torch.nn import functional as F

from .audio import convert_audio_channels
from .compressed import get_musdb_tracks

import pandas as pd
from sklearn.model_selection import train_test_split

MIXTURE = "mix_clean"
EXT = ".wav"


def _track_metadata(track):
    track_length = None
    track_samplerate = None
    for idx,source in enumerate(track[:-1]):
        #file = track / f"{source}{EXT}"
        file = Path(source)
        info = ta.info(str(file))
        length = info.num_frames
        if track_length is None:
            track_length = length
            track_samplerate = info.sample_rate
        elif track_length != length:
            raise ValueError(
                f"Invalid length for file {file}: "
                f"expecting {track_length} but got {length}.")
        elif info.sample_rate != track_samplerate:
            raise ValueError(
                f"Invalid sample rate for file {file}: "
                f"expecting {track_samplerate} but got {info.sample_rate}.")
        if idx == 0:
            wav, _ = ta.load(str(file))
            wav = wav.mean(0)
            mean = wav.mean().item()
            std = wav.std().item()

    return {"length": length, "mean": mean, "std": std, "samplerate": track_samplerate}


def _build_metadata(path):
    meta_train = {}
    meta_valid = {}
    meta_test = {}
    path = Path(path)

    # First, generate train and validation metadata
    metadata_train_valid_csv = path / "metadata" / "mixture_train-100_mix_clean.csv" # hardcoded for now
    df = pd.read_csv(metadata_train_valid_csv)
    df=df.to_numpy()
    n_rows = df.shape[0]
    train_size=0.95 # hardcoded for now
    train_rows,valid_rows = train_test_split(range(n_rows),train_size=train_size)
    
    # Generate train metadata
    i=0
    for row in train_rows: #range(df.shape[0])
        ID = df[row,0]
        meta_train[ID]=_track_metadata(df[row,1:])
        i+=1
        if i % 100:
            print(i)

    # Generate valid metadata
    for row in valid_rows: #range(df.shape[0])
        ID = df[row,0]
        meta_valid[ID]=_track_metadata(df[row,1:])
        i+=1
        if i % 100:
            print(i)

    # Next, generate test metadata
    metadata_test_csv = path / "metadata" / "mixture_test_mix_clean.csv" # hardcoded for now
    df = pd.read_csv(metadata_test_csv)
    df=df.to_numpy()
    n_rows = df.shape[0]
    
    # Generate test metadata
    for row in range(n_rows): #range(df.shape[0])
        ID = df[row,0]
        meta_test[ID]=_track_metadata(df[row,1:])
        i+=1
        if i % 100:
            print(i)
            
    return meta_train, meta_valid, meta_test




    #print(f"Path in _build_metadata function: {path}")
    #for root, folders, files in os.walk(path, followlinks=True):
    #    root = Path(root)
    #    if root.name.startswith('.') or folders or root == path:
    #        continue
    #    name = str(root.relative_to(path))
    #    meta[name] = _track_metadata(root, sources)
    #return meta


class Wavset:
    def __init__(
            self,
            root, metadata, sources,
            length=None, stride=None, normalize=True,
            samplerate=44100, channels=2,is_test=False):
        """
        Waveset (or mp3 set for that matter). Can be used to train
        with arbitrary sources. Each track should be one folder inside of `path`.
        The folder should contain files named `{source}.{ext}`.
        Files will be grouped according to `sources` (each source is a list of
        filenames).

        Sample rate and channels will be converted on the fly.

        `length` is the sample size to extract (in samples, not duration).
        `stride` is how many samples to move by between each example.
        """
        self.root = Path(root)
        self.metadata = OrderedDict(metadata)
        self.length = length
        self.stride = stride or length
        self.normalize = normalize
        self.sources = sources
        self.channels = channels
        self.samplerate = samplerate
        self.num_examples = []
        self.is_test=is_test

        for name, meta in self.metadata.items():
            track_length = int(self.samplerate * meta['length'] / meta['samplerate'])
            if length is None or track_length < length:
                examples = 1
            else:
                examples = int(math.ceil((track_length - self.length) / self.stride) + 1)
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def get_file(self, name, source):
        if self.is_test:
            return self.root / "test" / source / f"{name}{EXT}" 
        else:
            return self.root / "train-100" / source / f"{name}{EXT}"

    def __getitem__(self, index):
        for name, examples in zip(self.metadata, self.num_examples):
        
            if index >= examples:
                index -= examples
                continue
            meta = self.metadata[name]
            num_frames = -1
            offset = 0
            if self.length is not None:
                offset = int(math.ceil(
                    meta['samplerate'] * self.stride * index / self.samplerate))
                num_frames = int(math.ceil(
                    meta['samplerate'] * self.length / self.samplerate))
            wavs = []
            for source in self.sources:
                file = self.get_file(name, source)
                wav, _ = ta.load(str(file), frame_offset=offset, num_frames=num_frames)
                wav = convert_audio_channels(wav, self.channels)
                wavs.append(wav)

            example = th.stack(wavs)
            example = julius.resample_frac(example, meta['samplerate'], self.samplerate)
            if self.normalize:
                example = (example - meta['mean']) / meta['std']
            if self.length:
                example = example[..., :self.length]
                example = F.pad(example, (0, self.length - example.shape[-1]))
            return example


def get_wav_datasets(args, samples, sources):
    sig = hashlib.sha1(str(args.wav).encode()).hexdigest()[:8]
    metadata_file = args.metadata / (sig + ".json")
    #metadata_file = args.metadata / ("mixture_train-100_mix_clean" + ".csv") 
    #train_path = args.wav # input should be: E:\Libri2Mix\wav8k\max
    #valid_path = args.wav # input should be: E:\Libri2Mix\wav8k\max
    root = args.wav # input should be: E:\Libri2Mix\wav8k\max
    print(f"Root path: {root}")
    #print(f"Train path: {train_path}")
    #print(f"Validation path: {valid_path}")
    if not metadata_file.is_file() and args.rank == 0:
        metadata_train, metadata_valid, metadata_test = _build_metadata(root)
        json.dump([metadata_train, metadata_valid, metadata_test], open(metadata_file, "w"))
        #train = _build_metadata(train_path, sources)
        #valid = _build_metadata(valid_path, sources)
        #json.dump([train, valid], open(metadata_file, "w"))
    if args.world_size > 1:
        distributed.barrier()
    metadata_train, metadata_valid, metadata_test = json.load(open(metadata_file))
    #train, valid = json.load(open(metadata_file))

    
    train_set = Wavset(root, metadata_train, sources,
                       length=samples, stride=args.data_stride,
                       samplerate=args.samplerate, channels=args.audio_channels,
                       normalize=args.norm_wav)
    valid_set = Wavset(root, metadata_valid, [MIXTURE] + sources,
                       samplerate=args.samplerate, channels=args.audio_channels,
                       normalize=args.norm_wav)
    
    test_set = Wavset(root, metadata_test, [MIXTURE] + sources,
                    samplerate=args.samplerate, channels=args.audio_channels,
                    normalize=args.norm_wav,is_test=True)

    return train_set, valid_set, test_set


def get_musdb_wav_datasets(args, samples, sources):
    metadata_file = args.metadata / "musdb_wav.json"
    root = args.musdb / "train"
    if not metadata_file.is_file() and args.rank == 0:
        metadata = _build_metadata(root, sources)
        json.dump(metadata, open(metadata_file, "w"))
    if args.world_size > 1:
        distributed.barrier()
    metadata = json.load(open(metadata_file))

    train_tracks = get_musdb_tracks(args.musdb, is_wav=True, subsets=["train"], split="train")
    metadata_train = {name: meta for name, meta in metadata.items() if name in train_tracks}
    metadata_valid = {name: meta for name, meta in metadata.items() if name not in train_tracks}
    train_set = Wavset(root, metadata_train, sources,
                       length=samples, stride=args.data_stride,
                       samplerate=args.samplerate, channels=args.audio_channels,
                       normalize=args.norm_wav)
    valid_set = Wavset(root, metadata_valid, [MIXTURE] + sources,
                       samplerate=args.samplerate, channels=args.audio_channels,
                       normalize=args.norm_wav)
    return train_set, valid_set
