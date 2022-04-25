import tqdm
from tqdm import tqdm
import json
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

# From https://stackoverflow.com/questions/42583670/how-to-add-new-elements-to-the-end-of-a-json-file
def append_to_json(filepath, data):
    """
    Append data in JSON format to the end of a JSON file.
    NOTE: Assumes file contains a JSON object (like a Python
    dict) ending in '}'. 
    :param filepath: path to file
    :param data: dict to append
    """

    # construct JSON fragment as new file ending
    new_ending = ", " + json.dumps(data)[1:-1] + "}\n"

    # edit the file in situ - first open it in read/write mode
    with open(filepath, 'r+') as f:

        f.seek(0, 2)        # move to end of file
        index = f.tell()    # find index of last byte

        # walking back from the end of file, find the index 
        # of the original JSON's closing '}'
        while not f.read().startswith('}'):
            index -= 1
            if index == 0:
                raise ValueError("can't find JSON object in {!r}".format(filepath))
            f.seek(index)

        # starting at the original ending } position, write out
        # the new ending
        f.seek(index)
        f.write(new_ending)    