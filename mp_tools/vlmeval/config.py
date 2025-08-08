import os
from vlmeval.vlm import *
from functools import partial
from vlmeval.vlm.gen_id_tspo import GEN_Frame_ID_TSPO

supported_VLM = {
    "TSPO": partial(  
        GEN_Frame_ID_TSPO,
        model_path="/your/path/TSPO-0.4B", 
        root='/your/path/open_TSPO',
        save_root='/your/path/video_feats',
        sample_num=64,
    ), 
}

