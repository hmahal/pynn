from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from facelandmarksdataset import FaceLandmarksDataset

import warnings
warnings.filterwarnings("ignore")

plt.ion()

def show_landmarks(image, landmarks):
  """Show images with the landmarks"""
  plt.imshow(image)
  plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
  plt.pause(0.001)


face_dataset = FaceLandmarksDataset(csv_file='faces/face_landmarks.csv', root_dir='faces/') 
fig = plt.figure()

for i in range(len(face_dataset)):
  sample = face_dataset[i]

  print(i, sample['image'].shape, sample['landmarks'].shape)

  ax = plt.subplot(1, 4, i + 1)
  plt.tight_layout()
  ax.set_title('Sample #{}'.format(i))
  ax.axis('off')
  show_landmarks(**sample)

  if i == 3:
    plt.show()
    break

input("wait")