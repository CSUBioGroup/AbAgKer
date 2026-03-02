# training base model
nohup python main_wandb.py --gpu_nodes 1 --base any_tests/config_1120/AbAgKer_fold1_1120.yaml -t True --gpus 0, > log_runout/AbAgKer_fold1_1120_test61.log 2>&1 &
nohup python main_wandb.py --gpu_nodes 1 --base any_tests/config_1120/AbAgKer_fold2_1120.yaml -t True --gpus 0, > log_runout/AbAgKer_fold2_1120_test62.log 2>&1 &
nohup python main_wandb.py --gpu_nodes 1 --base any_tests/config_1120/AbAgKer_fold3_1120.yaml -t True --gpus 0, > log_runout/AbAgKer_fold3_1120_test63.log 2>&1 &
nohup python main_wandb.py --gpu_nodes 1 --base any_tests/config_1120/AbAgKer_fold4_1120.yaml -t True --gpus 0, > log_runout/AbAgKer_fold4_1120_test64.log 2>&1 &
nohup python main_wandb.py --gpu_nodes 1 --base any_tests/config_1120/AbAgKer_fold5_1120.yaml -t True --gpus 0, > log_runout/AbAgKer_fold5_1120_test65.log 2>&1 &


#  AbAgI enhance
nohup python main_wandb.py --gpu_nodes 1 --base any_tests/config_1120_withAbAgIdata/AbAgKer_fold1_1120_withAbAgIdata.yaml -t True --gpus 0, > log_runout/AbAgKer_fold1_1120_withAbAgIdata_test66.log 2>&1 &
nohup python main_wandb.py --gpu_nodes 1 --base any_tests/config_1120_withAbAgIdata/AbAgKer_fold2_1120_withAbAgIdata.yaml -t True --gpus 0, > log_runout/AbAgKer_fold2_1120_withAbAgIdata_test67.log 2>&1 &
nohup python main_wandb.py --gpu_nodes 1 --base any_tests/config_1120_withAbAgIdata/AbAgKer_fold3_1120_withAbAgIdata.yaml -t True --gpus 0, > log_runout/AbAgKer_fold3_1120_withAbAgIdata_test68.log 2>&1 &
nohup python main_wandb.py --gpu_nodes 1 --base any_tests/config_1120_withAbAgIdata/AbAgKer_fold4_1120_withAbAgIdata.yaml -t True --gpus 0, > log_runout/AbAgKer_fold4_1120_withAbAgIdata_test69.log 2>&1 &
nohup python main_wandb.py --gpu_nodes 1 --base any_tests/config_1120_withAbAgIdata/AbAgKer_fold5_1120_withAbAgIdata.yaml -t True --gpus 0, > log_runout/AbAgKer_fold5_1120_withAbAgIdata_test70.log 2>&1 &

# koff predictor
nohup python main_wandb.py --gpu_nodes 1 --base any_tests/config_1120_koff/AbAgKer_fold1_1120_Koff.yaml -t True --gpus 0, > log_runout/AbAgKer_fold1_1120_Koff_test76.log 2>&1 &
nohup python main_wandb.py --gpu_nodes 1 --base any_tests/config_1120_koff/AbAgKer_fold2_1120_Koff.yaml -t True --gpus 0, > log_runout/AbAgKer_fold2_1120_Koff_test77.log 2>&1 &
nohup python main_wandb.py --gpu_nodes 1 --base any_tests/config_1120_koff/AbAgKer_fold3_1120_Koff.yaml -t True --gpus 0, > log_runout/AbAgKer_fold3_1120_Koff_test78.log 2>&1 &
nohup python main_wandb.py --gpu_nodes 1 --base any_tests/config_1120_koff/AbAgKer_fold4_1120_Koff.yaml -t True --gpus 0, > log_runout/AbAgKer_fold4_1120_Koff_test79.log 2>&1 &
nohup python main_wandb.py --gpu_nodes 1 --base any_tests/config_1120_koff/AbAgKer_fold5_1120_Koff.yaml -t True --gpus 0, > log_runout/AbAgKer_fold5_1120_Koff_test80.log 2>&1 &



