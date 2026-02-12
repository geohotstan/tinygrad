import unittest, sys
import numpy as np
from tinygrad import Tensor, dtypes, Context
from tinygrad.helpers import CI, Profiling

@unittest.skipIf(sys.platform.startswith("win"), "flaky on Windows")
class TestWinograd(unittest.TestCase):
  def _assert_forward_close(self, x:Tensor, w:Tensor, *, atol=1e-4, **kwargs):
    with Context(WINO=0, SCACHE=0):
      expected = Tensor.conv2d(x, w, **kwargs).realize()
    with Context(WINO=1, SCACHE=0):
      result = Tensor.conv2d(x, w, **kwargs).realize()
    np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=atol, rtol=1e-4)

  def _assert_backward_close(self, x_shape:tuple[int, ...], w_shape:tuple[int, ...], *, seed=1337, atol=1e-4, **kwargs):
    def run(new_wino:int):
      Tensor.manual_seed(seed)
      x = Tensor.randn(*x_shape, requires_grad=True)
      w = Tensor.randn(*w_shape, requires_grad=True)
      with Context(WINO=new_wino, SCACHE=0):
        out = Tensor.conv2d(x, w, **kwargs)
        loss = out.square().mean()
        loss.backward()
      return x.grad.realize().numpy(), w.grad.realize().numpy()

    xg_expected, wg_expected = run(0)
    xg_result, wg_result = run(1)
    np.testing.assert_allclose(xg_result, xg_expected, atol=atol, rtol=1e-4)
    np.testing.assert_allclose(wg_result, wg_expected, atol=atol, rtol=1e-4)

  def test_profile(self):
    x,w = Tensor.rand(1,4,9,9).realize(), Tensor.rand(4,4,3,3).realize()
    with Profiling(enabled=not CI, sort='time'):
      with Context(WINO=1, SCACHE=0):
        Tensor.conv2d(x,w).realize()

  def test_schedule_switches_with_new_wino(self):
    x, w = Tensor.randn(1, 4, 9, 9).realize(), Tensor.randn(4, 4, 3, 3).realize()
    with Context(WINO=0, SCACHE=0):
      baseline = Tensor.conv2d(x, w, padding=1).schedule()
    with Context(WINO=1, SCACHE=0):
      new_wino = Tensor.conv2d(x, w, padding=1).schedule()
    self.assertNotEqual(tuple(si.ast.key for si in baseline), tuple(si.ast.key for si in new_wino))

  def test_schedule_fallback_matches_baseline(self):
    x, w = Tensor.randn(1, 4, 13, 13).realize(), Tensor.randn(6, 4, 5, 5).realize()
    with Context(WINO=0, SCACHE=0):
      baseline = Tensor.conv2d(x, w, padding=2).schedule()
    with Context(WINO=1, SCACHE=0):
      new_wino = Tensor.conv2d(x, w, padding=2).schedule()
    self.assertEqual(tuple(si.ast.key for si in baseline), tuple(si.ast.key for si in new_wino))

  def test_forward_matches_baseline(self):
    cases = [
      {"x": (1, 4, 9, 9), "w": (4, 4, 3, 3), "kwargs": {"padding": 1}},
      {"x": (2, 6, 13, 9), "w": (6, 2, 3, 3), "kwargs": {"groups": 3, "padding": (1, 0, 2, 1)}},
      {"x": (2, 4, 17), "w": (6, 4, 3), "kwargs": {"padding": 1}},
      {"x": (1, 4, 7, 6, 5), "w": (8, 2, 3, 3, 3), "kwargs": {"groups": 2, "padding": 1}, "atol": 2e-4},
    ]
    for i, case in enumerate(cases):
      with self.subTest(case=i):
        x = Tensor.randn(*case["x"]).realize()
        w = Tensor.randn(*case["w"]).realize()
        self._assert_forward_close(x, w, atol=case.get("atol", 1e-4), **case["kwargs"])

  def test_backward_matches_baseline(self):
    self._assert_backward_close((1, 4, 9, 9), (4, 4, 3, 3), padding=1)

  def test_fallback_matches_baseline(self):
    cases = [
      {"x": (1, 4, 13, 13), "w": (6, 4, 5, 5), "kwargs": {"padding": 2}},
      {"x": (1, 6, 15, 11), "w": (6, 2, 3, 3), "kwargs": {"groups": 3, "padding": 1, "stride": 2}},
      {"x": (1, 4, 17), "w": (5, 4, 3), "kwargs": {"padding": 2, "dilation": 2}},
    ]
    for i, case in enumerate(cases):
      with self.subTest(case=i):
        x = Tensor.randn(*case["x"]).realize()
        w = Tensor.randn(*case["w"]).realize()
        self._assert_forward_close(x, w, **case["kwargs"])

  def test_dtype(self):
    IC, OC, X, Y = 4,4,9,9
    x,w = Tensor.empty(1,IC,Y,X), Tensor.empty(OC,IC,3,3)
    with Context(WINO=1, SCACHE=0):
      self.assertEqual(Tensor.conv2d(x,w).dtype, dtypes.default_float)

    x,w = Tensor.empty(1,IC,Y,X,dtype=dtypes.half), Tensor.empty(OC,IC,3,3,dtype=dtypes.half)
    with Context(WINO=1, SCACHE=0):
      self.assertEqual(Tensor.conv2d(x,w).dtype, dtypes.half)

if __name__ == '__main__':
  unittest.main(verbosity=2)
