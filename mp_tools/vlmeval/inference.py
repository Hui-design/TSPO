import os
import time
from datetime import datetime
import json
from vlmeval.utils import listinstr
from vlmeval.dataset import split_MMMU


class _Worker:
    """Function wrapper for ``track_progress_rich``"""

    def __init__(self, func) -> None:
        self.func = func

    def __call__(self, inputs):
        inputs, idx = inputs
        if not isinstance(inputs, (tuple, list, dict)):
            inputs = (inputs, )
        # import pdb; pdb.set_trace()
        inputs["index"] = idx
        if isinstance(inputs, dict):
            return self.func(**inputs), idx
        else:
            return self.func(*inputs), idx


def inference_data(model_name, structs, gpu_list_str, queue, dataset_build_prompt_func=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list_str
    import torch
    from vlmeval.config import supported_VLM
    model = supported_VLM[model_name]() if isinstance(
        model_name, str) else model_name
    worker = _Worker(model.generate)
    # structs = [
    #     dict(message=struct, dataset=dataset_name)
    #     for struct in structs
    # ]
    for item in structs:
        dataset_name = item["dataset"]
        struct = item["struct"]
        idx = struct["index"]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(struct, dataset=dataset_name)
        else:
            struct = dataset_build_prompt_func(struct, dataset_name)
        if listinstr(["MMMU"], dataset_name):
            struct = split_MMMU(struct)
        struct = {"message": struct, 'dataset': dataset_name}
        result, idx_ = worker((struct, idx))
        torch.cuda.empty_cache()
        queue.put((result, idx_))
    finished_info = {"status": "finished"}
    if hasattr(model, "model_meta"):
        finished_info["meta_data"] = model.model_meta
    else:
        finished_info["meta_data"] = {
            "Method": [model_name, "Tele"],
            "Parameters": "0B",
            "Language Model": "-",
            "Vision Model": "-",
            "Org": "Tele",
            # generating time string in format "YYYY/MM/DD"
            "Time": datetime.now().strftime("%Y/%m/%d"),
            "Verified": "Yes",
            "OpenSource": "Yes"
        }
    queue.put(json.dumps(finished_info))
