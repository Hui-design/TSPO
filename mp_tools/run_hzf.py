import os
import json
import datetime
import filelock
import os.path as osp
import argparse
from tqdm import tqdm
from collections import defaultdict
from vlmeval.utils import (
    load_env,
    get_logger,
    get_available_gpus,
    dump,
    load,
)
from vlmeval.inference import inference_data
from multiprocessing import Process, Queue
from jinja2 import Environment, FileSystemLoader
from vlmeval.dataset import split_MMMU, DATASET_TYPE, build_video

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    # Essential Args
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        default=[
            "LongVideoBench",
            "MLVU",
            "VideoMME",
            "LVBench"
        ],
    )
    parser.add_argument("--model", type=str, nargs="+", required=True)
    # Work Dir
    parser.add_argument(
        "--work-dir", type=str, default="./work_dir", help="select the output directory"
    )
    parser.add_argument("--gpus-per-worker", type=int,
                        default=1, help="GPU per worker")
    # Infer + Eval or Infer Only
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "infer"])
    # API Kwargs, Apply to API VLMs and Judge API LLMs
    parser.add_argument("--nproc", type=int, default=4,
                        help="Parallel API calling")
    parser.add_argument(
        "--retry", type=int, default=None, help="retry numbers for API VLMs"
    )
    # Explicitly Set the Judge Model
    parser.add_argument("--judge", type=str, default=None)
    parser.add_argument(
        '--is-api-model', action='store_true', help='is api model')
    # Logging Utils
    parser.add_argument("--verbose", action="store_true")

    # Configuration for Resume
    # Ignore: will not rerun failed VLM inference
    parser.add_argument("--ignore", action="store_true",
                        help="Ignore failed indices. ")
    # Rerun: will remove all evaluation temp files
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    return args

def main():
    logger = get_logger("RUN")
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    ps = []
    # ipdb.set_trace()
    work_dir = args.work_dir
    model_name = args.model[0]
    dataset_names = args.data
    for dataset_name in dataset_names:
        custom_flag = False

        pred_root = osp.join(args.work_dir, model_name)
        os.makedirs(pred_root, exist_ok=True)
        result_file = f"{pred_root}/{model_name}_{dataset_name}.json"
        if osp.exists(result_file) and args.rerun:
            os.system(f"rm {pred_root}/{model_name}_{dataset_name}_*")

        out_file = osp.join(work_dir, f"{model_name}_{dataset_name}_supp.pkl")
        res = load(out_file) if osp.exists(out_file) else defaultdict(None)
        meta_file = osp.join(pred_root, "meta.json")
        model_meta = load(meta_file) if osp.exists(meta_file) else {}

        dataset = build_video(dataset_name)
        data = dataset.data # 取1/3
        # import pdb; pdb.set_trace()
        lt, indices = len(data), list(data["index"])

        dataset_build_prompt_func = dataset.build_prompt
        structs = [data.iloc[i] for i in range(lt)]

        queue = Queue()
        structs = [s for s in structs if s["index"] not in res]
        import random  
        random.shuffle(structs) # modified by tch
        # import pdb; pdb.set_trace()
        if len(structs) > 0:
            if args.is_api_model:
                num_workers = args.nproc
                available_gpus = [str(idx)
                                  for idx in range(8)]
            else:
                available_gpus = [str(idx)
                                  for idx in get_available_gpus(10 * 1024)]
                # available_gpus = ["7"]
                num_workers = len(available_gpus) // args.gpus_per_worker
            # For now, we do not use split_MMMU for MMMU dataset
            structs = [
                dict(struct=struct, dataset=dataset_name)
                for struct in structs
            ]
            print("num_workers:", num_workers)
            tbar = tqdm(total=len(structs))
            total_chunks = len(structs)
            num_samples_per_worker = (len(structs) - 1) // num_workers + 1
            for i in range(num_workers):
                sub_structs = structs[
                    i * num_samples_per_worker: (i + 1) * num_samples_per_worker
                ]
                gpu_list_str = ",".join(
                    available_gpus[
                        i * args.gpus_per_worker: (i + 1) * args.gpus_per_worker
                    ]
                )
                # inference_data(model_name, sub_structs,
                #                gpu_list_str, queue, dataset_build_prompt_func)
                p = Process(
                    target=inference_data,
                    args=(model_name, sub_structs, gpu_list_str,
                          queue, dataset_build_prompt_func),
                )
                p.start()
                ps.append(p)

            num_finished = 0
            model_meta = {}
            finished_chunks = 0
            save_counter = 0
            while True:
                if finished_chunks >= total_chunks:
                    break
                item = queue.get()
                if isinstance(item, str):
                    num_finished += 1
                    finished_info = json.loads(item)
                    model_meta = finished_info["meta_data"]
                    if num_finished >= len(ps):
                        break
                else:
                    tbar.update(1)
                    result, idx = item
                    res[idx] = result
                    finished_chunks += 1
                    save_counter += 1
                    if save_counter >= 100:
                        if not os.path.exists(meta_file):
                            dump(model_meta, meta_file)
                        dump(res, out_file)
                        save_counter = 0

            # Final save for any remaining results
            if not os.path.exists(meta_file):
                dump(model_meta, meta_file)
            dump(res, out_file)

            for p in ps:
                p.join()
        # for x in data["index"]:
        #     assert x in res
        # data["prediction"] = [str(res[x]) for x in data["index"]]

        # dump(data, result_file)


if __name__ == "__main__":
    load_env()
    main()
