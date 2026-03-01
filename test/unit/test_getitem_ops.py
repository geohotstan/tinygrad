import unittest
import numpy as np
from tinygrad import Tensor, GlobalCounters

# NOTE: btw global_ops are 0 for both these tests

class TestGetitemOps(unittest.TestCase):
  def test_broadcasted_outer_indices_stays_two_stage(self):
    src_np = np.arange(16*32, dtype=np.int32).reshape(16, 32)
    ib_np, jb_np = np.array([[1], [3], [7], [9]], dtype=np.int32), np.array([[2, 4, 6]], dtype=np.int32)
    src, ib, jb = Tensor(src_np).realize(), Tensor(ib_np).realize(), Tensor(jb_np).realize()
    out = src[ib, jb]
    np.testing.assert_equal(out.numpy(), src_np[ib_np, jb_np])
    # row gather (4x32) then column gather (4x3)
    self.assertEqual(len(out.schedule()), 2)

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
