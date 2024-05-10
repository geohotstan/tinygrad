import numpy as np
from tqdm import trange
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import CI
from tinygrad.engine.jit import TinyJit


def train(model, X_train, Y_train, optim, steps, BS=128, lossfn=lambda out,y: out.sparse_categorical_crossentropy(y),
        transform=lambda x: x, target_transform=lambda x: x, noloss=False, allow_jit=True):

  def train_step(x, y):
    # network
    out = model.forward(x) if hasattr(model, 'forward') else model(x)
    loss = lossfn(out, y)
    optim.zero_grad()
    loss.backward()
    if noloss: del loss
    optim.step()
    if noloss: return (None, None)
    cat = out.argmax(axis=-1)
    accuracy = (cat == y).mean()
    return loss.realize(), accuracy.realize()

  if allow_jit: train_step = TinyJit(train_step)

  with Tensor.train():
    losses, accuracies = [], []
    for i in (t := trange(steps, disable=CI)):
      samp = Tensor.randint(BS, low=0, high=X_train.shape[0])
      x = transform(X_train[samp])
      x.requires_grad = False
      y = target_transform(Y_train[samp])
      loss, accuracy = train_step(x, y)
      # printing
      if not noloss:
        loss, accuracy = loss.numpy(), accuracy.numpy()
        losses.append(loss)
        accuracies.append(accuracy)
        t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))
  return [losses, accuracies]


def evaluate(model, X_test, Y_test, num_classes=None, BS=128, return_predict=False, transform=lambda x: x,
             target_transform=lambda y: y):
  Tensor.training = False
  def eval(Y_test, num_classes):
    # NOTE: contiguous for setitem
    Y_test_preds_out = Tensor.zeros(Y_test.shape+(num_classes,)).contiguous()
    for i in trange((Y_test.shape[0]-1)//BS+1, disable=CI):
      x = transform(X_test[i*BS:(i+1)*BS])
      out = model.forward(x) if hasattr(model, 'forward') else model(x)
      # HACK: is turning requires_grad off here right? it is needed for setitem
      out.requires_grad = False
      Y_test_preds_out[i*BS:(i+1)*BS] = out
    Y_test_preds = Tensor.argmax(Y_test_preds_out, axis=-1)
    Y_test = target_transform(Y_test)
    return (Y_test == Y_test_preds).mean().item(), Y_test_preds

  if num_classes is None: num_classes = (Y_test.max().cast(dtypes.int)+1).item()
  acc, Y_test_pred = eval(Y_test, num_classes)
  print("test set accuracy is %f" % acc)
  return (acc, Y_test_pred) if return_predict else acc