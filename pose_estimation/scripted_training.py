import sys

from train import train_loop, prepare_config, prepare_training_set
import os
import subprocess
import json
from tqdm import tqdm

mlflow_env = {'MLFLOW_TRACKING_URI': 'http://localhost:5000', 'MLFLOW_S3_ENDPOINT_URL': 'http://localhost:9000'}

for env in mlflow_env:
    os.environ[env] = mlflow_env[env]

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_registry_uri("http://localhost:9000")

experiment_count = 0

if len(sys.argv) != 2:
    print("Usage: python scripted_training.py PATH TO EXPERIMENT_CONFIG")
    sys.exit(1)

training_config_file = sys.argv[1]
with open(training_config_file) as file:
    training_config = json.load(file)

train_experiments = training_config['training_experiments']
training_data = training_config['training_data']

for key in train_experiments:
    experiment_count += len(train_experiments[key])
experiment_count *= len(training_data)

progress_bar = tqdm(total=experiment_count)

failed_experiments = []

for data_set in training_data:
    prepare_training_set('/home/jfeil/IDP/training_pipeline/data/tennis_dataset', data_set['train_set'],
                         data_set['val_set'], '/var/tmp/test')
    for experiment_id in train_experiments:
        for dataset_params, train_params in train_experiments[experiment_id]:
            with mlflow.start_run(experiment_id=experiment_id) as run:
                config_path = prepare_config('/var/tmp/test', 'experiment_output',
                                             '../experiments/coco/resnet152/384x288_d256x3_adam_lr1e-3_TrainingLoopDefault.yaml',
                                             dataset_params, train_params, deterministic=False)
                mlflow.log_params(data_set)
                result = subprocess.run(["python3", "train.py", "--cfg", config_path, "--mlflow-run", run.info.run_id],
                                        env=os.environ)
                if result.returncode != 0:
                    print(f"FAILED {(experiment_id, dataset_params, train_params)}")
                    failed_experiments += [(experiment_id, dataset_params, train_params)]
                    mlflow.end_run('FAILED')
                # train_loop(config_path, num_workers=16, enable_mlflow=False)
                os.remove(config_path)
                progress_bar.update(1)

if failed_experiments:
    print(failed_experiments)
