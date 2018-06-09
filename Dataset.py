import collections
import numpy as np
import pandas as pd
from tensorflow.python.framework import dtypes, random_seed

#定义图像调用函数
def load(data_file):
  data = pd.read_csv(data_file)
  pixels = data['pixels'].tolist()
  width, height = 48, 48
  faces = []
  for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(width, height)
    faces.append(face)
  faces = np.asarray(faces)
  faces = np.expand_dims(faces, -1)
  emotions = pd.get_dummies(data['emotion']).as_matrix()
  return faces, emotions

#定义数据集类，简化调入数据流程
class DataSet(object):
  def __init__(self,images,labels,reshape=True,dtype=dtypes.float32,seed=None):
    seed1, seed2 = random_seed.get_seed(seed)
    np.random.seed(seed1 if seed is None else seed2)
    if reshape:
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],images.shape[1]*images.shape[2])

    if dtype == dtypes.float32:
      images = images.astype(np.float32)
      images = np.multiply(images, 1.0 / 255.0)

    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self.num_examples

  @property
  def epochs_completed(self):
    self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    start = self._index_in_epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self._images[perm0]
      self._labels = self._labels[perm0]
    if start + batch_size > self._num_examples:
      self._epochs_completed += 1
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

def input_data(train_dir,dtype=dtypes.float32,reshape=True,seed=None):
  training_size = 23000
  validation_size = 3589

  train_faces, train_emotions = load(train_dir)
  print('Dataset load success!!')
  perm0 = np.arange(len(train_emotions))
  np.random.shuffle(perm0)

  train_faces = train_faces[perm0]
  train_emotions = train_emotions[perm0]

  # 测试集
  validation_faces = train_faces[training_size : training_size + validation_size]
  validation_emotions = train_emotions[training_size : training_size + validation_size]

  # 评估集
  test_faces = train_faces[training_size + validation_size : ]
  test_emotions = train_emotions[training_size + validation_size : ]

  # 训练集
  train_faces = train_faces[ : training_size]
  train_emotions = train_emotions[ : training_size]

  Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
  train = DataSet(train_faces, train_emotions, reshape=reshape, seed=seed)
  validation = DataSet(validation_faces, validation_emotions, dtype=dtype, reshape=reshape, seed=seed)
  test = DataSet(test_faces, test_emotions, dtype=dtype, reshape=reshape, seed=seed)
  return Datasets(train=train, validation=validation, test=test)

def utils():
  input= input_data('./data/fer2013/fer2013.csv')

if __name__ == '__main__':
  utils()
