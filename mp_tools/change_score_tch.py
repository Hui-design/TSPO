import argparse
import json
import os
from tqdm import tqdm
import pickle

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_anno_path', type=str, default='../evaluation')
    parser.add_argument('--data', type=str, choices=["VideoMME","LongVideoBench","MLVU"])
    parser.add_argument('--name', type=str, default='xx')
    args = parser.parse_args()
    return args

def load_jsonlines(jsonl_path):
    with open(jsonl_path, "r") as f:
        data = [json.loads(line) for line in tqdm(f, desc="Loading jsonl file")]
    return data

if __name__ == '__main__':
    args = parser_args()
    json_name = {"VideoMME":'videomme',"LongVideoBench":"lvb_val","MLVU":'mlvu'}[args.data]
    anno_path = os.path.join(args.base_anno_path, 'jsons', f'{json_name}.json')
    score_path = f"work_dir/{args.name}_{args.data}_supp.pkl"
    target_path = os.path.join(args.base_anno_path, 'jsons_idx', f'{args.name}_{args.data}_frameIdx.json')

    with open(anno_path, 'r') as f:
        anno = json.load(f) 
    with open(score_path, 'rb') as f:
        score = pickle.load(f)

    for idx, data in enumerate(anno):
        if args.data.lower() == "videomme" or args.data.lower() == "mlvu":
            index = data["question_id"]
        elif args.data.lower() == "longvideobench":
            index = data["id"]
        else:
            raise NotImplementedError("to be implemented")
        if index not in score:
            print(index)
            continue
        data["frame_idx"] = score[index]
    with open(target_path, 'w') as f:
        json.dump(anno, f)