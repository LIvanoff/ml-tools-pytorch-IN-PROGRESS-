import os
import pathlib
import pickle
import numpy as np
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder


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


def create_dataset(X_data,
                   permutate: bool,
                   workers: int,
                   batch_size: int,
                   augment: bool,
                   filetype: str,
                   image_size: int,
                   mode: str,
                   y_data=None
                   ):
    if y_data is not None and filetype:
        Xy_data = TableDataset(X_data, y_data)
    else:
        Xy_data = CVDataset(X_data, mode, RESCALE_SIZE=image_size, augment=augment)

    if mode == 'train':
        Xy_data = DataLoader(Xy_data, batch_size=batch_size, shuffle=permutate)
    else:
        Xy_data = DataLoader(Xy_data, batch_size=batch_size, shuffle=False)

    return Xy_data


class CVDataset(Dataset):
    def __init__(self, files, mode, RESCALE_SIZE, augment):
        super().__init__()
        self.files = sorted(files)
        self.mode = mode
        self.RESCALE_SIZE = RESCALE_SIZE
        self.augment = augment

        self.len_ = len(self.files)

        self.label_encoder = LabelEncoder()

        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            with open('label_encoder.pkl', 'wb') as le_dump_file:
                pickle.dump(self.label_encoder, le_dump_file)

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image

    def __getitem__(self, index):
        # для преобразования изображений в тензоры PyTorch и нормализации входа
        if self.mode == 'train':
            if not self.augment:
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(size=(self.RESCALE_SIZE, self.RESCALE_SIZE)),
                    transforms.RandomRotation(degrees=25),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        x = self.load_sample(self.files[index])
        x = self._prepare_sample(x)
        # x = np.array(x / 255, dtype='float32')
        x = transform(x)
        if self.mode == 'test':
            return x
        else:
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y

    def _prepare_sample(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)

        image = transforms.functional.pad(image, padding, 0, 'constant')
        # image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
        return np.array(image)


class TableDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X)
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return (self.X[index], self.y[index])
