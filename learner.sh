
export PYTHONPATH=$PYTHONPATH:../../lerobot/src/
export PYTHONPATH=$PYTHONPATH:../../../HIL-RL


task_name=close_trashbin_franka_1028

mkdir -p experiments/${task_name}
cd experiments/${task_name}

python3 ../../learner.py robot_type@_global_=franka task@_global_=${task_name} policy_type=silri

