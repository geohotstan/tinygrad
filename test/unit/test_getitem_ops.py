import unittest
import numpy as np
from tinygrad import Tensor, GlobalCounters
from tinygrad.helpers import Context

# NOTE: btw global_ops are 0 for both these tests

class TestGetitemOps(unittest.TestCase):
  @staticmethod
  def _get_fancy_index_tensors():
    # mirror TestOps._get_index_randoms with deterministic numpy RNG
    rng = np.random.default_rng(0)
    i = Tensor(rng.integers(-1, 1, size=(2,1,1,1,1,1), dtype=np.int32)).realize()
    j = Tensor(rng.integers(0, 1, size=(1,3,1,1,1,1), dtype=np.int32)).realize()
    k = Tensor(rng.integers(-5, 5, size=(1,1,4,1,1,1), dtype=np.int32)).realize()
    o = Tensor(rng.integers(0, 4, size=(2,1,1,5,1,1), dtype=np.int32)).realize()
    p = Tensor(rng.integers(0, 1, size=(1,1,1,1,6,1), dtype=np.int32)).realize()
    return i, j, k, o, p

  # @unittest.expectedFailure
  def test_slice_fancy_indexing_variations_single_kernel(self):
    with Context(PCONTIG=2):
      # Covers test/backend/test_ops.py TestOps.test_slice_fancy_indexing_* variations (excluding *_errors).
      src = Tensor(np.arange(2*5*6*5*3*4, dtype=np.float32).reshape(2, 5, 6, 5, 3, 4)).realize()
      i, j, k, o, p = self._get_fancy_index_tensors()

      cases: list[tuple[str, Tensor]] = [
        # no_dim_collapse
        ("no_dim_collapse_0", src[i,j,k,o,p]),
        ("no_dim_collapse_1", src[:,j,k,o,:]),
        ("no_dim_collapse_2", src[i,j,...]),
        ("no_dim_collapse_3", src[i,...,p]),
        ("no_dim_collapse_4", src[...,k,:,p]),

        # dim_collapse_int
        ("dim_collapse_int_0", src[1,j,k,o,p]),
        ("dim_collapse_int_1", src[i,j,3,o,p]),
        ("dim_collapse_int_2", src[1,j,2,o,2]),
        ("dim_collapse_int_3", src[i,2,2,2,p]),
        ("dim_collapse_int_4", src[1,:,3:11:2,o,0:2]),

        # dim_inject_none
        ("dim_inject_none_0", src[None,j,k,o,p]),
        ("dim_inject_none_1", src[i,j,k,o,None]),
        ("dim_inject_none_2", src[i,j,None,o,p]),
        ("dim_inject_none_3", src[None,j,k,o,None]),
        ("dim_inject_none_4", src[i,:,None,o,p]),
        ("dim_inject_none_6", src[None,None,j,k,o,p]),
        ("dim_inject_none_7", src[None,None,j,k,None,None]),
        ("dim_inject_none_8", src[i,None,None,k,o,p]),
        ("dim_inject_none_9", src[i,None,None,k,None,None]),
        ("dim_inject_none_10", src[None,None,j,None,o,p]),

        # dim_inject_and_collapse
        ("dim_inject_and_collapse_0", src[1,j,None,o,1]),
        ("dim_inject_and_collapse_1", src[None,j,2,o,None]),
        ("dim_inject_and_collapse_2", src[...,1,o,None]),
      ]

      src_small = Tensor(np.arange(2*3, dtype=np.float32).reshape(2, 3)).realize()
      cases += [
        ("with_tensors_0", src_small[Tensor([[0,0,0],[0,0,0]]).realize(), Tensor(1).realize()]),
        ("with_tensors_1", src_small[Tensor([1]).realize(), Tensor([[0,0,0],[0,0,0]]).realize()]),
        ("with_tensors_2", src_small[Tensor([[0,0,0],[0,0,0]]).realize(), Tensor([2,1,1]).realize()]),
        ("with_tensors_3", src_small[Tensor([[0,1,-1],[-1,-2,0]]).realize(), Tensor([2,1,-1]).realize()]),
      ]

      non_single_kernel: list[tuple[str, int]] = []
      for name, out in cases:
        kernels = len(out.schedule())
        if kernels != 1: non_single_kernel.append((name, kernels))
      self.assertEqual(non_single_kernel, [])

  def test_broadcasted_outer_indices_stays_two_stage(self):
    with Context(PCONTIG=2):
      src_np = np.arange(16*32, dtype=np.int32).reshape(16, 32)
      ib_np, jb_np = np.array([[1], [3], [7], [9]], dtype=np.int32), np.array([[2, 4, 6]], dtype=np.int32)
      src, ib, jb = Tensor(src_np).realize(), Tensor(ib_np).realize(), Tensor(jb_np).realize()
      out = src[ib, jb]
      np.testing.assert_equal(out.numpy(), src_np[ib_np, jb_np])
      # row gather (4x32) then column gather (4x3)
      self.assertEqual(len(out.schedule()), 1)

  def test_two_tensor_same_shape_indices_single_stage(self):
    src_np = np.arange(10*100*200, dtype=np.float32).reshape(10, 100, 200)
    idx1_np = np.array([[1, 3, 5], [2, 4, 6]], dtype=np.int32)
    idx2_np = np.array([[7, 9, 11], [8, 10, 12]], dtype=np.int32)
    src, idx1, idx2 = Tensor(src_np).realize(), Tensor(idx1_np).realize(), Tensor(idx2_np).realize()
    out = src[0, idx1, idx2]
    np.testing.assert_equal(out.numpy(), src_np[0, idx1_np, idx2_np])
    # Same-shape advanced indices should fuse to one stage.
    self.assertEqual(len(out.schedule()), 1)

  def test_two_tensor_indices(self):
    # linear indexing is O(idx_size), one-hot masks is O(idx_size * src_size)
    src_np = np.random.rand(10, 100, 200).astype(np.float32)
    idx1_np, idx2_np = np.random.randint(0, 100, (50, 60), dtype=np.int32), np.random.randint(0, 200, (50, 60), dtype=np.int32)
    src, idx1, idx2 = Tensor(src_np), Tensor(idx1_np), Tensor(idx2_np)
    # O(50*60) = 3K vs O(50*60*100*200) = 60M
    GlobalCounters.reset()
    np.testing.assert_allclose(src_np[0, idx1_np, idx2_np], src[0, idx1, idx2].numpy())
    self.assertLess(GlobalCounters.global_ops, 50_000)
    # consecutive indices not starting from dim 0: O(10*50*60) = 30K vs O(10*50*60*100*200) = 600M
    GlobalCounters.reset()
    np.testing.assert_allclose(src_np[:, idx1_np, idx2_np], src[:, idx1, idx2].numpy())
    self.assertLess(GlobalCounters.global_ops, 500_000)

  def test_two_tensor_indices_non_linear(self):
    # linear indexing is O(idx_size), one-hot masks is O(idx_size * src_size)
    src_np = np.random.rand(100, 10, 200).astype(np.float32)
    idx1_np, idx2_np = np.random.randint(0, 100, (50, 60), dtype=np.int32), np.random.randint(0, 200, (50, 60), dtype=np.int32)
    src, idx1, idx2 = Tensor(src_np), Tensor(idx1_np), Tensor(idx2_np)
    # O(50*60) = 3K vs O(50*60*100*200) = 60M
    GlobalCounters.reset()
    np.testing.assert_allclose(src_np[idx1_np, 0, idx2_np], src[idx1, 0, idx2].numpy())
    self.assertLess(GlobalCounters.global_ops, 50_000)
    # consecutive indices not starting from dim 0: O(10*50*60) = 30K vs O(10*50*60*100*200) = 600M
    GlobalCounters.reset()
    np.testing.assert_allclose(src_np[idx1_np, :, idx2_np], src[idx1, :, idx2].numpy())
    self.assertLess(GlobalCounters.global_ops, 500_000)

if __name__ == '__main__':
  unittest.main()
