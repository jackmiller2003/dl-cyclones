import tqdm
from tqdm import tqdm
import ujson as json
import logging
import os

def init_logger():
    logging.basicConfig(filename = "json.log",
                        filemode = "w",
                        format = "%(levelname)s %(asctime)s - %(message)s", 
                        level = logging.ERROR)
    return logging.getLogger()    

# Utility function for adding to a JSON
def add_to_json(url_key,new_data, file_name):
    if os.path.isfile(file_name):
        with open(file_name,'r+') as f:
            data = json.load(f)
            data[url_key] = new_data # <--- add `id` value.
            f.seek(0)        # <--- should reset file position to the beginning.
            json.dump(data, f, indent=4)
            f.truncate()     # remove remaining part

            # file_data = json.load(file, lines=True)
            # file_data[url_key] = new_data
            # json.dump(file_data, file, indent = 4)
    else:
        write_new_json(url_key,new_data,file_name)

# Writing a new JSON if it doesn't already exist
def write_new_json(url_key,new_data,file_name):
    with open(file_name, 'w') as json_file:
            json_dict = {}
            json_dict[url_key] = new_data
            json.dump(json_dict, json_file)

