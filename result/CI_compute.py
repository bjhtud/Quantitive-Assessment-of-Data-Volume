import json
import math
import os

import numpy as np
from scipy.stats import t

result_path = './result'

dataset_folders = [name for name in os.listdir(result_path)
           if os.path.isdir(os.path.join(result_path, name))]
for dataset_name in dataset_folders:
    dataset_result_path = os.path.join(result_path, dataset_name)
    test_size_folders = [name for name in os.listdir(dataset_result_path)
                       if os.path.isdir(os.path.join(dataset_result_path, name))]
    for test_size in test_size_folders:
        test_size_path_path = os.path.join(dataset_result_path, test_size)
        seed_folders = [name for name in os.listdir(test_size_path_path)
                       if os.path.isdir(os.path.join(test_size_path_path, name))]
        r2_list = []
        for seed in seed_folders:
            seed_path = os.path.join(test_size_path_path, seed)
            with open(os.path.join(seed_path, f'result.json'), 'r') as f:
                result = json.load(f)
            r2 = result['R2_score']
            r2_list.append(r2)

        for alpha in [0.05, 0.1, 0.2]:
            t_value_size = t.ppf(1 - alpha / 2, int(test_size))
            CI_test = t_value_size * 0.5 / math.sqrt(int(test_size))
            t_value_seeds = t.ppf(1 - alpha / 2, 30)
            CI_seed = (t_value_seeds * np.std(r2_list))/ math.sqrt(30)
            bottom_r2 = np.mean(r2_list) - CI_test - CI_seed

            result_LCL = {
                'alpha': alpha,
                'LCL': bottom_r2
            }

            with open(os.path.join(test_size_path_path, f'CI_{(1-alpha)*100}%.json'), 'w') as f:
                json.dump(result_LCL, f)

