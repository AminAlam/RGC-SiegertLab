import pickle as pkl
import json

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pkl.load(f)

def load_sorting_data(sorting_file_path):
    with open(sorting_file_path) as f:
        sorting_file = json.load(f)
        paradigm = sorting_file['paradigms']
        traces = sorting_file['traces']
        experiment_name = sorting_file['experiment_name']
        route_name = sorting_file['route_name']
    return paradigm, traces, experiment_name, route_name