import math
import matplotlib.pyplot as plt
from autosklearn.regression import AutoSklearnRegressor
# from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from scipy.stats import t
from sklearn.metrics import r2_score
from utils import data_process, data_process_meta
import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test_size', type=float, default=20)
    parser.add_argument('--run_time', type=int, default=15)
    parser.add_argument('--dataset_name', type=str, default='Xiong_2014')

    args = parser.parse_args()
    seed = int(args.seed)
    test_size = float(args.test_size)/100
    run_time = int(args.run_time)
    dataset_name = args.dataset_name
    if dataset_name == 'Xiong_2014' or dataset_name == 'KIproBatt':
        test_fixed = True
    else:
        test_fixed = False
    # read dataset
    X_train, X_test, y_train, y_test = data_process_meta('./dataset/meta.csv', dataset_name=dataset_name,
                                                         random_state=seed, test_size=test_size, scale=False, test_fixed=test_fixed)

    # redefine test_size
    if dataset_name == 'Xiong_2014' or dataset_name == 'KIproBatt':
        test_size = int(len(X_test)/(len(X_train) + len(X_test)) * 100)

    # get current working directory
    current_dir = os.getcwd()

    # create tmp folder and result folder
    tmp_folder_path = os.path.join(current_dir, 'tmp')
    if not os.path.exists(tmp_folder_path):
        os.makedirs(tmp_folder_path, exist_ok=True)

    tmp_folder_path = tmp_folder_path + f'/autosklearn_regression_example_tmp_{dataset_name}_{seed}_{test_size}_{run_time}'

    result_path = './result'
    result_path = os.path.join(result_path, dataset_name)
    result_path = os.path.join(result_path, str(test_size))
    result_path = os.path.join(result_path, str(seed))

    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)

    # model training
    automl = AutoSklearnRegressor(
        time_left_for_this_task=run_time * 60,
        per_run_time_limit=run_time * 5,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5},
        n_jobs=-1,
        delete_tmp_folder_after_terminate=True,
        tmp_folder=tmp_folder_path,
    )

    automl.fit(X_train, y_train)

    # model evaluation
    y_pred = automl.predict(X_test)
    R2_score = r2_score(y_test, y_pred)

    result = {
        'job_id': os.environ.get('SLURM_JOB_ID', 'local'),
        'hyperparameters': {'seed': seed, 'test_size': test_size, 'run_time': run_time},
        'R2_score': R2_score,
    }
  
    # save result

    poT = automl.performance_over_time_
    poT.plot(
        x="Timestamp",
        kind="line",
        legend=True,
        title="Auto-sklearn accuracy over time",
        grid=True,
    )
    plt.savefig(os.path.join(result_path, 'performance_over_time.png'))
    with open(os.path.join(result_path, f'result.json'), 'w') as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()