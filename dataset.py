'''
    This code is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This code is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this code.  If not, see <https://www.gnu.org/licenses/>.

    Copyright (c) Saige Research
    All rights reserved.
'''

import pickle
import torch
from torch.utils import data
from torchvision import transforms


class sv_dataset(data.Dataset):
    def __init__(self, split):
        self.split = split
        self.load_image_list()
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def load_image_list(self):
        if self.split == "train":
            with open("data_train.pkl", "rb") as f:
                self.dataset = pickle.load(f)
        elif self.split == "validation":
            with open("data_validation.pkl", "rb") as f:
                self.dataset = pickle.load(f)
        else:
            raise NotImplementedError(f"{self.split} is not a proper split name")

    def __getitem__(self, index):

        img = self.dataset[index]
        img = self.to_tensor(img)

        return img