import unittest
from tinygrad import dtypes
from tinygrad.codegen.lowerer import pm_push_half_cast
from tinygrad.ops import KernelInfo, UOp, Ops, graph_rewrite, PatternMatcher, UPat, sint, sint_to_uop, GroupOp

    # g1 = UOp(Ops.DEFINE_GLOBAL, dtypes.int32.ptr(), (), 0)
    # c1 = UOp(Ops.CONST, dtypes.int, (), 2)
    # c2 = UOp(Ops.CONST, dtypes.int, (), 3)
    # a1 = UOp(Ops.MUL, dtypes.int, (l1, c1))
    # a2 = UOp(Ops.MUL, dtypes.int, (l1, c2))

class TestPushCast(unittest.TestCase):
  def test_simple(self):
    r = UOp(Ops.REDUCE_AXIS, dtypes.float32, )
    g1 = UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), (), 0)
    c1 = UOp(Ops.CONST, dtypes.half, (), 2)
    c2 = UOp(Ops.CONST, dtypes.half, (), 3)
    l1 = UOp(Ops.LOAD, dtypes.half, (g1.index(c1),))
    l1.cast(dtypes.half)
