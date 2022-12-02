def _get_env_settings(env_name, dataset):
    if env_name == 'maze2d' and dataset == 'umaze':
        env_targets = [50] 
        scale = 1000. 
        start, end = [1,1], [3,1]
    elif env_name == 'maze2d' and dataset == 'medium':
        env_targets = [50]
        scale = 1000.
        start, end = [1,1], [6,6]
    elif env_name == 'maze2d' and dataset == 'large':
        env_targets = [50]  
        scale = 1000. 
        start, end = [1,1], [7,9]
    elif env_name == 'pen':
        env_targets = [3000]
        scale = 1000.
    elif env_name == 'door':
        env_targets = [3000]
        scale = 1000.
    elif env_name == 'hammer':
        env_targets = [13000]
        scale = 1000.
    elif env_name == 'relocate':
        env_targets = [5000]
        scale = 1000.
    else:
        raise NotImplementedError
    return env_targets, scale

def _get_env_dataset(env_name, dataset):
    if env_name == 'maze2d':
        dataset_path = f'path/2/{env_name}-{dataset}-v1.pkl' # CHANGE ME! 
        validation_dataset_path = f'path/2/{env_name}-{dataset}-v1-validation.pkl' # CHANGE ME! 
    elif env_name == 'door' or env_name == 'pen' or env_name == 'hammer' or env_name == "relocate":
        dataset_path = f'path/2/{env_name}-{dataset}-v1.pkl' # CHANGE ME! 
        validation_dataset_path = f'path/2/{env_name}-v1-validation.pkl' # CHANGE ME! 
    else:
        raise NotImplementedError
    return dataset_path, validation_dataset_path

def _get_weights_directory(env_name, dataset):
    if env_name == 'maze2d':
        directory = f"weights/maze/{dataset}/"
    elif env_name == 'door' or env_name == 'pen' or env_name == 'hammer' or env_name == "relocate":
        directory = f"weights/adroit/{env_name}/"
    else:
        raise NotImplementedError
    return directory
    