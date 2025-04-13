import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_process(file_path, target_columns, target_to_fit, random_state=36, test_size=0.2, scale=False, ):
    # 分割数据集，返回训练集和验证集，重设默认索引
    data = pd.read_csv(file_path)
    data = data.dropna()
    X = data.drop(columns=target_columns)
    y = data[target_to_fit]
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    X_t, X_val, y_t, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # 重置索引
    X_t = X_t.reset_index(drop=True)  # drop=True 会丢弃旧的索引
    y_t = y_t.reset_index(drop=True)  # 同样丢弃旧的索引

    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    return X_t, X_val, y_t, y_val

def data_process_meta(file_path, dataset_name = None, random_state=36, start_row=0,test_size=0.2, scale=False, test_fixed = False):
    datasets = {}
    meta_data = pd.read_csv(file_path)
    meta_data = meta_data.iloc[start_row:]

    if dataset_name:

        meta_data.set_index('dataset_name', inplace=True)
        target_to_fit = meta_data.loc[dataset_name]['target_to_fit'].split(';')
        target_columns = meta_data.loc[dataset_name]['target_columns'].split(';')
        if test_fixed:
            paths = meta_data.loc[dataset_name]['path'].split(';')
            assert len(paths) == 2
            train, test = pd.read_csv(paths[0]), pd.read_csv(paths[1])
            X_t, X_val, y_t, y_val = train.drop(columns=target_columns), test.drop(columns=target_columns), train[target_to_fit], test[target_to_fit]
        else:
            path = meta_data.loc[dataset_name]['path']
            X_t, X_val, y_t, y_val = data_process(path, target_columns = target_columns, target_to_fit = target_to_fit, random_state = random_state,
                                                  test_size = test_size, scale = scale)
        return X_t, X_val, y_t, y_val

    else:
        for index, row in meta_data.iterrows():
            target_columns = row['target_columns'].split(';')
            target_to_fit = row['target_to_fit'].split(';')
            X_t, X_val, y_t, y_val = data_process(row['path'], target_columns = target_columns, target_to_fit = target_to_fit, random_state = random_state,
                                                  test_size = test_size, scale = scale)
            datasets[row['dataset_name']] = [X_t, X_val, y_t, y_val]


