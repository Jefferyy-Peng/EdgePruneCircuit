import os
import pickle

import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import pandas as pd

def color_grayscale_arr(arr, red=True):
  """Converts grayscale image to either red or green"""
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  if red:
    arr = np.concatenate([arr,
                          np.zeros((h, w, 2), dtype=dtype)], axis=2)
  else:
    arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                          arr,
                          np.zeros((h, w, 1), dtype=dtype)], axis=2)
  return arr
class ColoredMNIST(datasets.VisionDataset):
  """
  Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

  Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
  """
  def __init__(self, root='./data', env='train1', transform=None, target_transform=None, select_class=None):
    super(ColoredMNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform)

    self.prepare_colored_mnist()
    if env in ['train1', 'train2', 'test']:
      with open(os.path.join(self.root, 'ColoredMNIST', env) + '.p', 'rb') as file:
        self.data_label_tuples = pickle.load(file)
    elif env == 'all_train':
      with open(os.path.join(self.root, 'ColoredMNIST', 'train1.p'), 'rb') as file:
        train1 = pickle.load(file)
      with open(os.path.join(self.root, 'ColoredMNIST', 'train2.p'), 'rb') as file:
        train2 = pickle.load(file)
      self.data_label_tuples = train1 + train2
      if select_class is not None and select_class != 'all':
        filtered = [(img, label) for img, label in self.data_label_tuples if label == select_class]
        self.data_label_tuples = filtered
    elif env == 'all_train_unbiased':
      self.prepare_colored_mnist_unbiased()
      with open(os.path.join(self.root, 'ColoredMNIST', 'train1_unbiased.p'), 'rb') as file:
        train1 = pickle.load(file)
      with open(os.path.join(self.root, 'ColoredMNIST', 'train2_unbiased.p'), 'rb') as file:
        train2 = pickle.load(file)
      self.data_label_tuples = train1 + train2
      if select_class is not None and select_class != 'all':
        filtered = [(img, label) for img, label in self.data_label_tuples if label == select_class]
        self.data_label_tuples = filtered
    else:
      raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test, and all_train')

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.data_label_tuples)

  def prepare_colored_mnist(self):
    colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
    if os.path.exists(os.path.join(colored_mnist_dir, 'train1.p')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train2.p')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'test.p')):
      print('Colored MNIST dataset already exists')
      return

    print('Preparing Colored MNIST')
    train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

    train1_set = []
    train2_set = []
    test_set = []
    for idx, (im, label) in enumerate(train_mnist):
      if idx % 10000 == 0:
        print(f'Converting image {idx}/{len(train_mnist)}')
      im_array = np.array(im)

      # Assign a binary label y to the image based on the digit
      binary_label = 0 if label < 5 else 1

      # Flip label with 25% probability
      # if np.random.uniform() < 0.25:
      #   binary_label = binary_label ^ 1

      # Color the image either red or green according to its possibly flipped label
      color_red = binary_label == 0

      # Flip the color with a probability e that depends on the environment
      if idx < 20000:
        # 20% in the first training environment
        if np.random.uniform() < 0.2:
          color_red = not color_red
      elif idx < 40000:
        # 10% in the first training environment
        if np.random.uniform() < 0.1:
          color_red = not color_red
      else:
        # 90% in the test environment
        if np.random.uniform() < 0.9:
          color_red = not color_red

      colored_arr = color_grayscale_arr(im_array, red=color_red)

      if idx < 20000:
        train1_set.append((Image.fromarray(colored_arr), binary_label))
      elif idx < 40000:
        train2_set.append((Image.fromarray(colored_arr), binary_label))
      else:
        test_set.append((Image.fromarray(colored_arr), binary_label))

      # Debug
      # print('original label', type(label), label)
      # print('binary label', binary_label)
      # print('assigned color', 'red' if color_red else 'green')
      # plt.imshow(colored_arr)
      # plt.show()
      # break

    os.makedirs(colored_mnist_dir, exist_ok=True)
    with open(os.path.join(colored_mnist_dir, 'train1.p'), 'wb') as file:
      pickle.dump(train1_set, file)
    with open(os.path.join(colored_mnist_dir, 'train2.p'), 'wb') as file:
      pickle.dump(train2_set, file)
    with open(os.path.join(colored_mnist_dir, 'test.p'), 'wb') as file:
      pickle.dump(test_set, file)

  def prepare_colored_mnist_unbiased(self):
    colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
    if os.path.exists(os.path.join(colored_mnist_dir, 'train1_unbiased.p')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train2_unbiased.p')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'test_unbiased.p')):
      print('unbiased Colored MNIST dataset already exists')
      return

    print('Preparing unbiased Colored MNIST')
    train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

    train1_set = []
    train2_set = []
    test_set = []
    for idx, (im, label) in enumerate(train_mnist):
      if idx % 10000 == 0:
        print(f'Converting image {idx}/{len(train_mnist)}')
      im_array = np.array(im)

      # Assign a binary label y to the image based on the digit
      binary_label = 0 if label < 5 else 1

      # Flip label with 25% probability
      # if np.random.uniform() < 0.25:
      #   binary_label = binary_label ^ 1

      # Color the image either red or green according to its possibly flipped label
      color_red = binary_label == 0

      # Flip the color with a probability e that depends on the environment
      if idx < 20000:
        # 20% in the first training environment
        if np.random.uniform() < 0.5:
          color_red = not color_red
      elif idx < 40000:
        # 10% in the first training environment
        if np.random.uniform() < 0.5:
          color_red = not color_red
      else:
        # 90% in the test environment
        if np.random.uniform() < 0.5:
          color_red = not color_red

      colored_arr = color_grayscale_arr(im_array, red=color_red)

      if idx < 20000:
        train1_set.append((Image.fromarray(colored_arr), binary_label))
      elif idx < 40000:
        train2_set.append((Image.fromarray(colored_arr), binary_label))
      else:
        test_set.append((Image.fromarray(colored_arr), binary_label))

      # Debug
      # print('original label', type(label), label)
      # print('binary label', binary_label)
      # print('assigned color', 'red' if color_red else 'green')
      # plt.imshow(colored_arr)
      # plt.show()
      # break

    os.makedirs(colored_mnist_dir, exist_ok=True)
    with open(os.path.join(colored_mnist_dir, 'train1_unbiased.p'), 'wb') as file:
      pickle.dump(train1_set, file)
    with open(os.path.join(colored_mnist_dir, 'train2_unbiased.p'), 'wb') as file:
      pickle.dump(train2_set, file)
    with open(os.path.join(colored_mnist_dir, 'test_unbiased.p'), 'wb') as file:
      pickle.dump(test_set, file)

class WaterbirdDataset(Dataset):
  def __init__(self, data_correlation, split, root_dir, transform):
    self.split_dict = {
      'train': 0,
      'val': 1,
      'test': 2
    }
    self.env_dict = {
      (0, 0): 0,
      (0, 1): 1,
      (1, 0): 2,
      (1, 1): 3
    }
    self.split = split
    self.root_dir = root_dir
    self.dataset_name = "waterbird_complete" + "{:0.2f}".format(data_correlation)[-2:] + "_forest2water2"
    self.dataset_dir = os.path.join(self.root_dir, self.dataset_name)
    if not os.path.exists(self.dataset_dir):
      raise ValueError(
        f'{self.dataset_dir} does not exist yet. Please generate the dataset first.')
    self.metadata_df = pd.read_csv(
      os.path.join(self.dataset_dir, 'metadata.csv'))
    self.metadata_df = self.metadata_df[self.metadata_df['split'] == self.split_dict[self.split]]
    self.y_array = self.metadata_df['y'].values
    self.place_array = self.metadata_df['place'].values
    self.filename_array = self.metadata_df['img_filename'].values
    self.transform = transform
    self.target_name = 'y_array'

  def __len__(self):
    return len(self.filename_array)

  def filter_from_list(self, filter_list):
    self.y_array = self.y_array[filter_list]
    self.place_array = self.place_array[filter_list]
    self.filename_array = self.filename_array[filter_list]

  def __getitem__(self, idx):
    y = self.y_array[idx]
    place = self.place_array[idx]
    img_filename = os.path.join(
      self.dataset_dir,
      self.filename_array[idx])
    img = Image.open(img_filename).convert('RGB')
    img = self.transform(img)

    return img, y, place