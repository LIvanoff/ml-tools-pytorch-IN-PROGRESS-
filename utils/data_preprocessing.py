import os
import pathlib
from pathlib import Path


def check_dataset(dataset_path, root, sep):
    datasets_folder = Path(os.path.join(root, 'datasets'))
    msg = ''

    if not datasets_folder.exists():
        msg += "No such directory: 'datasets'"
        raise FileNotFoundError(msg)
        # os.makedirs(datasets_folder)

    num_files = count_files(datasets_folder)

    if num_files == 0:
        msg += "No files and folders in the datasets folder: add file or folder(s) in the datasets"
        raise FileNotFoundError(msg)
    elif num_files > 1 and dataset_path == '':
        msg += "Too many files in the datasets folder: select one of them"
        raise FileNotFoundError(msg)

    filename = dataset_path
    dataset_path = Path(os.path.join(datasets_folder, dataset_path))

    if not dataset_path.exists():
        msg += f"No such file: {dataset_path}"
        raise FileNotFoundError(msg)

    if dataset_path.is_file():
        if '.xlsx' or '.csv' or '.txt' in filename:
            if sep is None:
                msg += "Missing 1 required positional argument: 'sep'"
                raise TypeError(msg)
            else:
                filetype = pathlib.Path(dataset_path).suffix
                return [dataset_path], filetype
        else:
            # filename_list = filename.split('.')[-1]
            # filetype = filename_list[-1]
            msg += f"Extension *.{filename.split('.')[-1]} not implemented"
            raise NotImplementedError(msg)
    else:
        is_table = False
        return [dataset_path], is_table

    # if sep is not None:
    #     file_name = dataset_path
    #     if num_files > 1 and file_name == '':
    #         msg += "Too many files in the datasets folder: select one of them"
    #         raise FileNotFoundError(msg)
    #     elif file_name != '':

    # for i in os.listdir(datasets_folder):
    #     print(i)


def count_files(dataset_folder):
    folder_array = os.scandir(dataset_folder)
    num_files = 0
    for path in folder_array:
        if path.is_file():
            num_files += 1
    return num_files
