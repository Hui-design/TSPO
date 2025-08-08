import os
import string
import numpy as np
import hashlib
import os.path as osp
import warnings

import pandas as pd

from ..utils import (
    istype,
    download_file,
    LMUDataRoot,
    file_size,
    load,
    listinstr,
    toliststr,
    LOCALIZE,
    read_ok,
    decode_base64_to_image_file,
    decode_base64_to_image,
)
from .config import DATASET_TYPE, dataset_md5_dict, dataset_URLs, img_root_map


def check_md5(data_path, dataset):
    if dataset not in dataset_md5_dict:
        warnings.warn(
            f"We do not have an md5 record for dataset {dataset}, skip the md5 check. "
        )
        return True
    assert osp.exists(data_path)
    with open(data_path, "rb") as f:
        hash = hashlib.new("md5")
        for chunk in iter(lambda: f.read(2**20), b""):
            hash.update(chunk)
    if str(hash.hexdigest()) == dataset_md5_dict[dataset]:
        return True
    else:
        warnings.warn(
            "this data file is incomplete, so it needs to be downloaded again."
        )
        return False


def prep_tsv(dataset):
    data_root = LMUDataRoot()
    assert osp.exists(data_root)
    update_flag = False

    if dataset in dataset_URLs:
        url = dataset_URLs[dataset]
        file_name = url.split("/")[-1]
        data_path = osp.join(data_root, file_name)

        if osp.exists(data_path) and check_md5(data_path, dataset):
            pass
        else:
            warnings.warn("The dataset tsv is not downloaded")
            download_file(url, data_path)
            update_flag = True
    else:
        data_path = osp.join(data_root, dataset + ".tsv")
        assert osp.exists(data_path)

    if file_size(data_path, "GB") > 1:
        local_path = data_path.replace(".tsv", "_local.tsv")
        if (
            not osp.exists(local_path)
            or update_flag
            or os.environ.get("FORCE_LOCAL", None)
        ):
            LOCALIZE(data_path, local_path)
        return local_path
    else:
        return data_path


class VideoDataset:
    TYPE = "Video"

    def __init__(self, dataset="IntentQA", skip_noimg=True):
        self.data_root = LMUDataRoot()
        self.dataset = dataset
        self.dataset_type = DATASET_TYPE(dataset)
        self.data_path = prep_tsv(dataset)
        data = load(self.data_path)
        self.skip_noimg = skip_noimg

        # Prompt for Captioning
        if listinstr(["COCO"], dataset):
            data["question"] = [
                (
                    "Please describe this image in general. Directly provide the description, "
                    'do not include prefix like "This image depicts". '
                )
            ] * len(data)

        data["index"] = [str(x) for x in data["index"]]

        if "image_path" in data:
            paths = [toliststr(x) for x in data["image_path"]]
            data["image_path"] = [x[0] if len(x) == 1 else x for x in paths]

        if np.all([istype(x, int) for x in data["index"]]):
            data["index"] = [int(x) for x in data["index"]]

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return dict(self.data.iloc[idx])

    def build_prompt(self, line, dataset=None):
        if dataset is None:
            dataset = self.dataset

        if isinstance(line, int):
            line = self.data.iloc[line]

        ROOT = LMUDataRoot()
        video_root = osp.join(
            ROOT,
            "videos",
            img_root_map[dataset] if dataset in img_root_map else dataset,
            'video',
            'data',  # modified by tch
        )
        os.makedirs(video_root, exist_ok=True)
        tgt_path = toliststr(osp.join(video_root, str(line["video_name"])))

        prompt = line["question"]
        if DATASET_TYPE(dataset) == "multi-choice":
            question = line["question"]
            options = line['candidates']
            import ast
            # import ipdb; ipdb.set_trace()
            try:
                options = ast.literal_eval(options)
            except:
                print(line)
                assert False
            options_prompt = "Options:\n"
            for idx, candidate in enumerate(options):
                choice = chr(ord("A") + idx)
                options_prompt += f"({choice}):{candidate} "
            hint = (
                line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
            )
            prompt = ""
            if hint is not None:
                prompt += f"Hint: {hint}\n"
            prompt += f"Question: {question}\n"
            if len(options):
                prompt += options_prompt
                prompt += "Please select the correct answer from the options above. \n"
        elif DATASET_TYPE(dataset) == "VQA":
            if listinstr(["ocrvqa", "textvqa", "chartqa", "docvqa"], dataset.lower()):
                prompt += "\nAnswer the question using a single word or phrase.\n"

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type="video", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="video", value=tgt_path)]
        msgs.append(dict(type="text", value=prompt))

        return msgs

    def display(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        mmqa_display(line)
