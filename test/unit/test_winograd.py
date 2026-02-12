import unittest, sys
import numpy as np
from tinygrad import Tensor, Context, nn

@unittest.skipIf(sys.platform.startswith("win"), "flaky on Windows")
class TestWinogradClose(unittest.TestCase):
  def _assert_forward_close(self, x:Tensor, w:Tensor, *, bias:Tensor|None=None, atol=1e-4, **kwargs):
    with Context(WINO=0, SCACHE=0):
      expected = Tensor.conv2d(x, w, bias=bias, **kwargs).realize()
    with Context(WINO=1, SCACHE=0):
      result = Tensor.conv2d(x, w, bias=bias, **kwargs).realize()
    np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=atol, rtol=1e-4)

  def _assert_forward_backward_close(self, x_np:np.ndarray, w_np:np.ndarray, *, bias_np:np.ndarray|None=None, atol=1e-4, **kwargs):
    def run(new_wino:int):
      x = Tensor(x_np.copy(), requires_grad=True)
      w = Tensor(w_np.copy(), requires_grad=True)
      b = Tensor(bias_np.copy(), requires_grad=True) if bias_np is not None else None
      with Context(WINO=new_wino, SCACHE=0):
        out = Tensor.conv2d(x, w, bias=b, **kwargs)
        loss = out.square().mean()
        loss.backward()
      return out.realize().numpy(), x.grad.realize().numpy(), w.grad.realize().numpy(), (b.grad.realize().numpy() if b is not None else None)

    expected, xg_expected, wg_expected, bg_expected = run(0)
    result, xg_result, wg_result, bg_result = run(1)
    np.testing.assert_allclose(result, expected, atol=atol, rtol=1e-4)
    np.testing.assert_allclose(xg_result, xg_expected, atol=atol, rtol=1e-4)
    np.testing.assert_allclose(wg_result, wg_expected, atol=atol, rtol=1e-4)
    if bg_expected is not None and bg_result is not None:
      np.testing.assert_allclose(bg_result, bg_expected, atol=atol, rtol=1e-4)

  def test_close(self):
    inp = Tensor.rand(1, 16, 16, 16)
    conv = nn.Conv2d(16, 16, 3)
    with Context(WINO=0, SCACHE=0):
      cmp = conv(inp).realize()
    with Context(WINO=1, SCACHE=0):
      test = conv(inp).realize()
    np.testing.assert_allclose(cmp.numpy(), test.numpy(), atol=1e-5)

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

  def test_forward_generic_configs(self):
    cases = [
      {"x": (1, 3, 11, 28), "w": (4, 3, 3, 3), "kwargs": {"padding": 1}},
      {"x": (2, 6, 13, 9), "w": (6, 2, 3, 3), "kwargs": {"groups": 3, "padding": (1, 0, 2, 1)}},
      {"x": (2, 4, 17), "w": (6, 4, 3), "kwargs": {"padding": 1}},
      {"x": (1, 4, 7, 6, 5), "w": (8, 2, 3, 3, 3), "kwargs": {"groups": 2, "padding": 1}, "atol": 2e-4},
    ]
    for i, case in enumerate(cases):
      with self.subTest(case=i):
        x = Tensor.randn(*case["x"]).realize()
        w = Tensor.randn(*case["w"]).realize()
        self._assert_forward_close(x, w, atol=case.get("atol", 1e-4), **case["kwargs"])

  def test_forward_fallback_configs(self):
    cases = [
      {"x": (1, 4, 13, 13), "w": (6, 4, 5, 5), "kwargs": {"padding": 2}},
      {"x": (2, 6, 15, 11), "w": (6, 2, 3, 3), "kwargs": {"groups": 3, "padding": 1, "stride": 2}},
      {"x": (1, 4, 17), "w": (5, 4, 3), "kwargs": {"padding": 2, "dilation": 2}},
      {"x": (1, 2, 8, 8, 8), "w": (3, 2, 3, 3, 3), "kwargs": {"padding": 2, "dilation": 2}},
    ]
    for i, case in enumerate(cases):
      with self.subTest(case=i):
        x = Tensor.randn(*case["x"]).realize()
        w = Tensor.randn(*case["w"]).realize()
        self._assert_forward_close(x, w, **case["kwargs"])

  def test_forward_backward_generic_configs(self):
    cases = [
      {"x": (2, 4, 9, 9), "w": (6, 4, 3, 3), "bias": (6,), "kwargs": {"padding": 1}},
      {"x": (1, 6, 11, 8), "w": (6, 2, 3, 3), "bias": (6,), "kwargs": {"groups": 3, "padding": (1, 0, 2, 1)}},
      {"x": (1, 4, 13), "w": (5, 4, 3), "bias": (5,), "kwargs": {"padding": 1}},
      {"x": (1, 4, 7, 6, 5), "w": (8, 2, 3, 3, 3), "bias": (8,), "kwargs": {"groups": 2, "padding": 1}},
    ]
    for i, case in enumerate(cases):
      with self.subTest(case=i):
        x_np = Tensor.randn(*case["x"]).numpy().astype(np.float32, copy=True)
        w_np = Tensor.randn(*case["w"]).numpy().astype(np.float32, copy=True)
        b_np = Tensor.randn(*case["bias"]).numpy().astype(np.float32, copy=True)
        self._assert_forward_backward_close(x_np, w_np, bias_np=b_np, **case["kwargs"])

@unittest.skipIf(sys.platform.startswith("win"), "flaky on Windows")
class TestWinograd(unittest.TestCase):
  def test_padded_conv2d(self):
    # tests padding order in winograd
    x,w = Tensor.rand(1,3,11,28).realize(), Tensor.rand(4,3,3,3).realize()
    with Context(WINO=0, SCACHE=0): expected = Tensor.conv2d(x,w,padding=1).realize()
    with Context(WINO=1, SCACHE=0): result = Tensor.conv2d(x,w,padding=1).realize()
    np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)

if __name__ == '__main__':
  unittest.main(verbosity=2)
