#!/bin/sh
# # oc 配置火山云 aksk
# volc configure
# volc ml_task submit --conf volc_infer.yaml --entrypoint "nvidia-smi" --task_name test


# Usage
# bash lanch_oc_eval.sh 
HELP_MESSAGE="Usage: bash $0 [--help] [-w|--work_dir work_dir] [--user user] [-q|--queue_name queue_name] [--max_eval_step max_eval_step]"
# work_dir="/fs-computility/llm/lishuaibin/0813/marl_gen_original_amask/trainlog_2024-08-14-03:24:16"
# user="lishuaibin"
max_eval_step=400
queue_name="llmit"

while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            echo "$HELP_MESSAGE"; exit 1
            ;;
        -w|--work_dir)
            work_dir=$2; shift; shift
            ;;
        --user)
            user=$2; shift; shift
            ;;
        -q|--queue_name)
            queue_name=$2; shift; shift
            ;;
        --max_eval_step)
            max_eval_step=$2; shift; shift
            ;;
        *)
            echo "Unsupported positional argument: $1"; exit 1
            ;;
    esac
done

function assert_not_empty {
    if [ ! $# -eq 1 ]
    then
        echo "got $# arguments, expected 1"
        exit 1
    fi
    VARIABLE_NAME=$1
    if [ -z ${!VARIABLE_NAME} ]
    then
        echo "$VARIABLE_NAME is empty. Exit"
        exit 1
    fi
}

assert_not_empty work_dir
assert_not_empty user


###################################### 
# 以下配置无需修改
# conda activate
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate /fs-computility/llm/shared/llmeval/share_envs/oc-v030-ld-v052

which python
echo "user: $user"
echo "home: $HOME"

# get modify config
oc_out_dir="${work_dir}/oc_outputs"
mkdir -p ${oc_out_dir}
echo "oc 评测结果路径 ${oc_out_dir}"
cp -r /fs-computility/llm/shared/lishuaibin/oc_utils/corebench_v1_8 ${oc_out_dir}

volcano_config_path=${oc_out_dir}/corebench_v1_8/volc_infer.yaml
obj_config_path=${oc_out_dir}/corebench_v1_8/eval_chat_objective_volc.py
sub_config_path=${oc_out_dir}/corebench_v1_8/eval_chat_subjective_volc.py

# 修改home
sed -i "s/lishuaibin/${user}/" ${volcano_config_path}
sed -i "/.*home_path = ''*/c\home_path = '$HOME'" $sub_config_path
sed -i "/.*home_path = ''*/c\home_path = '$HOME'" $obj_config_path

# 修改 queue
if [ "$queue_name" = "llmit" ];then
    # llmit
    ResourceQueueID="q-20240425172512-t6mxc"
elif [ "$queue_name" = "llmit" ];then
    # hsllm_dd1
    ResourceQueueID="q-20240812151300-zwjbr"
fi

sed -i "s/''/${ResourceQueueID}/" ${volcano_config_path}
sed -i "/.*volcano_config_path = ''*/c\volcano_config_path = '${volcano_config_path}'" $sub_config_path
sed -i "/.*volcano_config_path = ''*/c\volcano_config_path = '${volcano_config_path}'" $obj_config_path
sed -i "s/queue_name = ''/queue_name = '${queue_name}'/" $sub_config_path
sed -i "s/queue_name = ''/queue_name = '${queue_name}'/" $obj_config_path


# get current models
ckpt_dirs="${work_dir}/ckpt/policy_model"

while [ ! -d "${ckpt_dirs}" ];
do
    echo "${ckpt_dirs} not exist, waiting..."
    sleep 5400
done

echo "从 ${ckpt_dirs} 获取模型并启动评测 ......"


cur_step=0
cur_model_dirs=()

while [[ $cur_step -le $max_eval_step ]];
do
    for file2 in `ls -a $ckpt_dirs`    
    do   
        if [ x"$file2" != x"." -a x"$file2" != x".." ];then

            if [ -d "$ckpt_dirs/$file2" ] && [[ ! " ${cur_model_dirs[@]} " =~ " $file2 " ]] ;then
                cur_model_dirs[${#cur_model_dirs[*]}]=$file2
                if [[ $cur_step -le ${file2} ]];then
                    cur_step=$((${file2}+1))
                fi

                tmp_path=$ckpt_dirs/$file2
                tmp_abbr="ppo_$file2"
                tmp_num_gpus=1
                # 添加待测模型到config
                tmp_model_cfg="('$tmp_abbr', '$tmp_path/', $tmp_num_gpus),"
                sed -i "53a ${tmp_model_cfg}" $sub_config_path
                sed -i "160a ${tmp_model_cfg}" $obj_config_path                

                # 提交评测
                nohup opencompass $obj_config_path -r latest -w ${oc_out_dir}/results > ${oc_out_dir}/objective.log 2>&1 &
                nohup opencompass $sub_config_path -r latest -w ${oc_out_dir}/results > ${oc_out_dir}/subjective.log 2>&1 &

                echo "评测新的模型 ${tmp_model_cfg} sleep 1.5h 等待评测结果以及下一个ckpt......"
                if [[ $cur_step -ge $max_eval_step ]];then
                    break
                fi
                sleep 5400 # sleep 1.5h for next model
            fi   
        fi    
    done
done


echo "================训练结束==========="
echo "评测模型: ${#cur_model_dirs[*]}, ${cur_model_dirs[*]}"
echo "评测结果：${oc_out_dir}"
