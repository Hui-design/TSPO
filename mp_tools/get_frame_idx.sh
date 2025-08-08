## dataset_name in [LongVideoBench, MLVU, VideoMME, LVBench]
export LMUData="evaluation/data"
dataset_name=$1
experiment_name=$2

pkl_path=work_dir/${experiment_name}_${dataset_name}_supp.pkl
if [ -f ${pkl_path} ]; then
    rm ${pkl_path}
    echo "The file has been deleted"
else
    echo "The file does not exist"
fi

python run_hzf.py --model ${experiment_name} --data ${dataset_name}
python change_score_tch.py --data ${dataset_name} --name ${experiment_name}
