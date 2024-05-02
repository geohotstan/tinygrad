#!/usr/bin/env python3
import numpy as np
from PIL import Image

from tinygrad import Tensor, dtypes
from tinygrad.nn.state import get_parameters
from tinygrad.nn import optim
from tinygrad.nn.datasets import mnist
from tinygrad.helpers import getenv
from extra.training import train, evaluate
from extra.models.resnet import ResNet


class ComposeTransforms:
  def __init__(self, trans):
    self.trans = trans

  def __call__(self, x):
    for t in self.trans:
      x = t(x)
    return x

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = mnist()
  X_train = X_train.reshape(-1, 28, 28).cast(dtypes.uint8)
  X_test = X_test.reshape(-1, 28, 28).cast(dtypes.uint8)
  classes = 10

  TRANSFER = getenv('TRANSFER')
  model = ResNet(getenv('NUM', 18), num_classes=classes)
  if TRANSFER:
    model.load_from_pretrained()

  lr = 5e-3
  transform = ComposeTransforms([
    # NOTE: numpy to apply Image transformation. Image.fromarray() requires obj to support array interface
    lambda x: [xx.numpy() for xx in x],
    lambda x: [Image.fromarray(xx, mode='L').resize((64, 64)) for xx in x],
    # change back to Tensor
    lambda x: [Tensor(np.asarray(xx)) for xx in x],
    lambda x: Tensor.stack(x, dim=0),
    lambda x: x / 255.0,
    lambda x: x.reshape(-1, 1).repeat([1,3,1,1]).cast(dtypes.float32)
  ])
  for _ in range(5):
    optimizer = optim.SGD(get_parameters(model), lr=lr, momentum=0.9)
    train(model, X_train, Y_train, optimizer, 100, BS=32, transform=transform)
    evaluate(model, X_test, Y_test, num_classes=classes, transform=transform)
    lr /= 1.2
    print(f'reducing lr to {lr:.7f}')
