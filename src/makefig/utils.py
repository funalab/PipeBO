import os
import json

def get_home_path():
    # Get path to parallel_boclass
    cwd = os.getcwd()
    return os.path.dirname(cwd)

def read_settings(json_f):
    with open(json_f, 'r') as f:
        jsn = json.load(f)
        batch_size = jsn['batch']
        param_num = []
        for process_key in jsn['parameter'].keys():
            param_num.append(len(jsn['parameter'][process_key]))
        param_num = tuple(param_num)
    return batch_size, param_num