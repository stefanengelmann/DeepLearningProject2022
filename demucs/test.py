# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import sys
from concurrent import futures

import musdb
import museval
import torch as th
import tqdm
from scipy.io import wavfile
from torch import distributed, nn


from .audio import convert_audio
from .utils import apply_model

from itertools import permutations
import numpy as np


def evaluate(model, 
             test_set,
             eval_folder,
             workers=2,
             device="cpu",
             rank=0,
             save=False,
             shifts=0,
             split=False,
             overlap=0.25,
             is_wav=False,
             world_size=1,
             is_mse=False):
    """
    Evaluate model using museval. Run the model
    on a single GPU, the bottleneck being the call to museval.
    """

    output_dir = eval_folder / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    json_folder = eval_folder / "results/test"
    json_folder.mkdir(exist_ok=True, parents=True)
    

    # we load tracks from the original musdb set
    #test_set = musdb.DB(musdb_path, subsets=["test"], is_wav=is_wav)
    #src_rate = 44100  # hardcoded for now...

    for p in model.parameters():
        p.requires_grad = False
        p.grad = None

    pendings = []
    with futures.ProcessPoolExecutor(workers or 1) as pool:
        for index in tqdm.tqdm(range(rank, len(test_set), world_size), file=sys.stdout):
            streams = test_set[index][0]
            # first five minutes to avoid OOM on --upsample models
            streams = streams[..., :15_000_000]
            streams = streams.to(device)
            mix = streams.sum(dim=0)
            references = streams
            mix = references.sum(dim=0)
            mean = test_set[index][1]
            std = test_set[index][2]
            name = test_set[index][3]

            out = json_folder / f"{name}.json.gz"
            if out.exists():
                continue

            #mix = th.from_numpy(mix).t().float()
            #ref = mix.mean(dim=0)  # mono mixture
            #mix = (mix - ref.mean()) / ref.std()
            #mix = convert_audio(mix, src_rate, model.samplerate, model.audio_channels)
            estimates = apply_model(model, mix.to(device),
                                    shifts=shifts, split=split, overlap=overlap)

            n_src=references.shape[0]
            perms = th.tensor(list(permutations(range(n_src))), dtype=th.long).to(device)

            if is_mse:
                testCriterion = nn.MSELoss()
            else:
                testCriterion = nn.L1Loss()

            testLoss=testCriterion(estimates,references)

            for perm in perms:
                estimates_tmp=estimates[perm,...]
                testLoss_tmp=testCriterion(estimates_tmp,references)
                if testLoss_tmp<testLoss:
                    testLoss=th.clone(testLoss_tmp)
                    estimates=th.clone(estimates_tmp)

            estimates = estimates * std + mean
            references = references * std + mean

            estimates = estimates.transpose(1, 2)
            # references = th.stack(
            #     [th.from_numpy(track.targets[name].audio).t() for name in model.sources])
            #references = convert_audio(references, src_rate,
            #                           model.samplerate, model.audio_channels)
            references = references.transpose(1, 2)
            references = references.cpu().numpy()
            estimates = estimates.cpu().numpy()
            win = int(1. * model.samplerate)
            hop = int(1. * model.samplerate)
            if save:
                folder = eval_folder / "wav/test" / name
                folder.mkdir(exist_ok=True, parents=True)
                for source, estimate in zip(model.sources, estimates):
                    wavfile.write(str(folder / (source + ".wav")), model.samplerate, estimate)

            if workers:
                pendings.append((name, pool.submit(
                    museval.evaluate, references, estimates, win=win, hop=hop)))
            else:
                pendings.append((name, museval.evaluate(
                    references, estimates, win=win, hop=hop)))
            del references, mix, estimates, streams

        for track_name, pending in tqdm.tqdm(pendings, file=sys.stdout):
            if workers:
                pending = pending.result()
            sdr, isr, sir, sar = pending
            track_store = museval.TrackStore(win=win, hop=hop, track_name=track_name)
            for idx, target in enumerate(model.sources):
                values = {
                    "SDR": sdr[idx].tolist(),
                    "SIR": sir[idx].tolist(),
                    "ISR": isr[idx].tolist(),
                    "SAR": sar[idx].tolist()
                }

                track_store.add_target(target_name=target, values=values)
                json_path = json_folder / f"{track_name}.json.gz"
                gzip.open(json_path, "w").write(track_store.json.encode('utf-8'))
    if world_size > 1:
        distributed.barrier()
