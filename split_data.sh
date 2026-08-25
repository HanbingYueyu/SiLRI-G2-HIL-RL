
export PYTHONPATH=$PYTHONPATH:../../../lerobot/src/
export PYTHONPATH=$PYTHONPATH:../../../RL-Robot-Env/
export PYTHONPATH=$PYTHONPATH:../../../HIL-RL

task_name=close_trashbin_franka_1028
mkdir -p experiments/${task_name}
cd experiments/${task_name}


python3 ../../split_data.py --repo_id=${task_name} --task=${task_name} --root=../../experiments/${task_name}/offline_dataset/${task_name} --output_root=../../experiments/${task_name}/offline_dataset/