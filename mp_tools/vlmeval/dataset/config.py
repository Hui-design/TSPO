from ..utils import listinstr

dataset_URLs = {
   
}

dataset_md5_dict = {
   
}

img_root_map = {k: k for k in dataset_URLs}
img_root_map.update({

})


def DATASET_TYPE(dataset):
    # Dealing with Custom Dataset
    if dataset is None:
        return None
    dataset = dataset.lower()
    if 'mmbench-video' in dataset:
        return 'VideoQA'
    elif listinstr([
        'mmbench', 'seedbench', 'ccbench', 'mmmu', 'scienceqa', 'ai2d',
        'mmstar', 'realworldqa', 'mmt-bench', 'aesbench', 'intentqa', 'videomme', 'egoschema', 'nextqa', 'longvideobench', 'mlvu', 'activitynetqa'
    ], dataset):
        return 'multi-choice'
    elif listinstr(['mme', 'hallusion', 'pope'], dataset) and not listinstr(['videomme'], dataset):
        return 'Y/N'
    elif 'coco' in dataset:
        return 'Caption'
    elif listinstr(['ocrvqa', 'textvqa', 'chartqa', 'mathvista', 'docvqa', 'infovqa', 'llavabench',
                    'mmvet', 'ocrbench', 'mllmguard'], dataset):
        return 'VQA'
    else:
        if dataset not in dataset_URLs:
            import warnings
            warnings.warn(f"Dataset {dataset} not found in dataset_URLs, will use 'multi-choice' as the default TYPE.")
            return 'multi-choice'
        else:
            return 'QA'


def abbr2full(s):
    datasets = [x for x in img_root_map]
    ins = [s in d for d in datasets]
    if sum(ins) == 1:
        for d in datasets:
            if s in d:
                return d
    else:
        return s
