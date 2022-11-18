import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import numpy as np
import os
import utils.func_utils as ut
from zipfile import ZipFile
import logging
from sklearn.model_selection import StratifiedShuffleSplit
import csv
import random
from sklearn.model_selection import StratifiedKFold


class CelebaDataset(Dataset):
    def __init__(self, list_IDs, labels, transform=T.ToTensor()):
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        img = Image.open(ID)
        X = self.transform(img)
        y = self.labels[ID]

        return X, y


def get_all_celeba_attributes():
    return ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
            'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
            'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
            'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
            'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
            'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
            'Wearing_Necktie', 'Young']


def get_all_fairface_attributes():
    return ['Gender', 'Race', 'Below_20', 'Below_30', 'Below_40']


def get_dataloader(target_attribute: int,
                   protected_attribute: int,
                   params: dict,
                   input_size: int = 224,
                   dataset: str = 'celebA',
                   augment: bool = False,
                   number_of_samples: int = 0,
                   split: str = 'train',
                   stratified: bool = False):
    if dataset == 'celebA':
        return get_celeba_dataloader(target_attribute=target_attribute,
                                     protected_attribute=protected_attribute,
                                     params=params,
                                     input_size=input_size,
                                     augment=augment,
                                     number_of_samples=number_of_samples,
                                     split=split,
                                     stratified=stratified)
    elif dataset == 'fairface':
        return get_fairface_dataloader(target_attribute_nmb=target_attribute,
                                       protected_attribute_nmb=protected_attribute,
                                       params=params,
                                       input_size=input_size,
                                       augment=augment,
                                       split=split,
                                       stratified=stratified)


def get_celeba_dataloader(target_attribute: int,
                          protected_attribute: int,
                          params: dict,
                          input_size: int = 224,
                          augment: bool = True,
                          number_of_samples: int = 0,
                          split: str = 'train',
                          stratified: bool = False):
    path = '../../data/celeba'
    img_path = path + '/img_align_celeba/'
    attr_path = 'data/celeba/list_attr_celeba.txt'

    if not os.path.isdir(img_path):
        ut.make_dir(path)
        logging.info('Extracting Files...')
        with ZipFile('data/celeba/img_align_celeba.zip', 'r') as zipObj:
            # Extract all the contents of zip file in different directory
            zipObj.extractall(path)

    list_ids = []
    label = open(attr_path, 'r')
    label = label.readlines()
    train_beg = 0
    valid_beg = 162770
    test_beg = 182637

    if split == 'train':
        end = valid_beg
        beg = train_beg
    elif split == 'valid':
        end = test_beg
        beg = valid_beg
    elif split == 'test':
        end = 202599
        beg = test_beg
    else:
        logging.info('Error')
        return
    if 0 < number_of_samples < end - beg:
        selected_images = random.sample(range(beg, end), number_of_samples)
    else:
        selected_images = range(beg, end)
    attr = {}
    positives_group_1 = 0
    positives_group_2 = 0
    group_1 = 0
    group_2 = 0
    for i in selected_images:
        temp = label[i+2].strip().split()
        list_ids.append(img_path + temp[0])
        attr[img_path + temp[0]] = torch.Tensor([int((int(temp[target_attribute + 1]) + 1) / 2), int((int(temp[protected_attribute + 1]) + 1) / 2)])
        if attr[img_path + temp[0]][1]:
            positives_group_1 += attr[img_path + temp[0]][0]
            group_1 += 1
        else:
            positives_group_2 += attr[img_path + temp[0]][0]
            group_2 += 1

    # These values are fixed by pytorch
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    resize_size = 256
    if input_size > 256:
        resize_size = 299
    if augment:
        transform = T.Compose([
            T.Resize(resize_size),
            T.RandomCrop(input_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
    else:
        transform = T.Compose([
            T.Resize(resize_size),
            T.CenterCrop(input_size),
            T.ToTensor(),
            normalize
        ])

    logging.info('Number of positives: {}'.format(positives_group_1 + positives_group_2))
    logging.info('Number of negatives: {}'.format(group_1 + group_2 - positives_group_1 - positives_group_2))
    logging.info('Number of group 1: {}'.format(group_1))
    logging.info('Number of group 2: {}'.format(group_2))
    logging.info('Positive Rate Group 1: {}'.format(positives_group_1 / group_1))
    logging.info('Positive Rate Group 2: {}'.format(positives_group_2 / group_2))

    if stratified:
        sampler = StratifiedBatchSampler(list_IDs=list_ids, labels=attr, batch_size=params['batch_size'])
        loader = DataLoader(CelebaDataset(list_ids, attr, transform), batch_sampler=sampler)
    else:
        loader = DataLoader(CelebaDataset(list_ids, attr, transform), **params)

    return loader


def get_fairface_dataloader(target_attribute_nmb: int,
                            protected_attribute_nmb: int,
                            params: dict,
                            input_size: int = 224,
                            augment: bool = True,
                            split: str = 'train',
                            stratified: bool = False):
    path = '../../data/fairface'
    img_path = path + '/fairface-img-margin125-trainval/'
    attr_path = 'data/fairface/'

    if not os.path.isdir(img_path):
        ut.make_dir(path)
        print('Extracting Files...')
        # Create a ZipFile Object and load sample.zip in it
        with ZipFile('data/fairface/fairface-img-margin125-trainval.zip', 'r') as zipObj:
            # Extract all the contents of zip file in different directory
            zipObj.extractall(path)

    attr_list = get_all_fairface_attributes()

    train_label_file = open(attr_path + 'fairface_label_train.csv')
    train_label_reader = csv.DictReader(train_label_file, delimiter=',')

    val_label_file = open(attr_path + 'fairface_label_val.csv')
    val_label_reader = csv.DictReader(val_label_file, delimiter=',')

    list_ids_train = []
    list_ids_val = []
    id_to_labels = {}
    mapping_age1 = {'0-2': 1, '3-9': 1, '10-19': 1, '20-29': 0, '30-39': 0,
                    '40-49': 0, '50-59': 0, '60-69': 0, 'more than 70': 0}
    mapping_age2 = {'0-2': 1, '3-9': 1, '10-19': 1, '20-29': 1, '30-39': 0,
                    '40-49': 0, '50-59': 0, '60-69': 0, 'more than 70': 0}
    mapping_age3 = {'0-2': 1, '3-9': 1, '10-19': 1, '20-29': 1, '30-39': 1,
                    '40-49': 0, '50-59': 0, '60-69': 0, 'more than 70': 0}
    mapping_gender = {'Male': 1, 'Female': 0}
    mapping_race2 = {'Black': 1, 'White': 0, 'East Asian': 1, 'Southeast Asian': 1,
                     'Latino_Hispanic': 1, 'Middle Eastern': 1, 'Indian': 1}

    mappings = {'Gender': (mapping_gender, 'gender'), 'Race': (mapping_race2, 'race'),
                'Below_20': (mapping_age1, 'age'), 'Below_30': (mapping_age2, 'age'),
                'Below_40': (mapping_age3, 'age')}
    for row in val_label_reader:
        list_ids_val.append(img_path + row['file'])
        prot_mapping, prot_col = mappings[attr_list[protected_attribute_nmb]]
        target_mapping, target_col = mappings[attr_list[target_attribute_nmb]]
        id_to_labels[img_path + row['file']] = torch.Tensor([target_mapping[row[target_col]], prot_mapping[row[prot_col]]])

    for row in train_label_reader:
        list_ids_train.append(img_path + row['file'])
        prot_mapping, prot_col = mappings[attr_list[protected_attribute_nmb]]
        target_mapping, target_col = mappings[attr_list[target_attribute_nmb]]
        id_to_labels[img_path + row['file']] = torch.Tensor([target_mapping[row[target_col]], prot_mapping[row[prot_col]]])

    labels_array = np.zeros(shape=(len(list_ids_train), 2))
    for idx, img_id in enumerate(list_ids_train):
        labels_array[idx] = id_to_labels[img_id]

    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.7, random_state=0)
    # split training/test name
    for train_idx, test_idx in sss.split(list_ids_train, labels_array):
        continue

    id_dict = {'train': np.array(list_ids_train)[train_idx],
                    'valid': np.array(list_ids_val),
                    'test': np.array(list_ids_train)[test_idx]}

    # These values are fixed by pytorch (following a discussion on github, no one really knows why anymore)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    resize_size = 256
    if input_size > 256:
        resize_size = 299
    if augment:
        transform = T.Compose([
            T.Resize(resize_size),
            T.RandomCrop(input_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
    else:
        transform = T.Compose([
            T.Resize(resize_size),
            T.CenterCrop(input_size),
            T.ToTensor(),
            normalize
        ])

    if stratified:
        sampler = StratifiedBatchSampler(list_IDs=id_dict[split], labels=id_to_labels, batch_size=params['batch_size'])
        loader = DataLoader(CelebaDataset(id_dict[split], id_to_labels, transform), batch_sampler=sampler)
    else:
        loader = DataLoader(CelebaDataset(id_dict[split], id_to_labels, transform), **params)

    return loader


class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    From https://discuss.pytorch.org/t/how-to-enable-the-dataloader-to-sample-from-each-class-with-equal-probability/911/7
    """

    def __init__(self, list_IDs, labels, batch_size, shuffle=True):
        self.sensitive_attr = []
        self.IDs = list_IDs
        for ID in list_IDs:
            self.sensitive_attr.append(labels[ID][-1])
        y = np.array(self.sensitive_attr)
        assert len(y.shape) == 1, 'label array must be 1D'
        self.n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=self.n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y), 1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0, int(1e8), size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return self.n_batches
