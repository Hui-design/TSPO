from .config import dataset_URLs, dataset_md5_dict, img_root_map, DATASET_TYPE, abbr2full
from .image_dataset import TSVDataset, split_MMMU
from .video_dataset import VideoDataset

def build_dataset(dataset_name, **kwargs):
    return TSVDataset(dataset_name, **kwargs)

def build_video(dataset_name, **kwargs):
    return VideoDataset(dataset_name, **kwargs)