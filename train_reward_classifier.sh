export PYTHONPATH=$PYTHONPATH:../../../lerobot/src/
export PYTHONPATH=$PYTHONPATH:../../../RL-Robot-Env/
export PYTHONPATH=$PYTHONPATH:../../../HIL-RL


# task_name=close_trashbin_franka_1028
task_name=fold_rag
# task_name=hang_clothes

mkdir -p experiments/${task_name}
cd experiments/${task_name}

# franka/ur singlearm task
python3 ../../train_reward_classifier.py \
    --config_path ../../cfg/train_config_reward_classifier.json \
    --dataset.root="../../experiments/${task_name}/offline_dataset/${task_name}" \
    "$@"  


# # tienkung dualarms task
# python3 ../../train_reward_classifier.py \
#     --config_path ../../cfg/train_config_reward_classifier_tienkung.json \
#     --dataset.root="../../experiments/${task_name}/offline_dataset/${task_name}" \
#     "$@"  