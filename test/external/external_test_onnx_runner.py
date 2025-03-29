# import unittest, onnx
# import numpy as np
# from tinygrad.frontend.onnx import OnnxRunner
#
# def helper_make_identity_model(inps: dict[str, np.ndarray], nm) -> onnx.ModelProto:
#   onnx_inputs = [onnx.helper.make_tensor_value_info(name, onnx.helper.np_dtype_to_tensor_dtype(arr.dtype), arr.shape) for name, arr in inps.items()]
#   onnx_outputs = [onnx.helper.make_tensor_value_info(name, onnx.helper.np_dtype_to_tensor_dtype(arr.dtype), arr.shape) for name, arr in inps.items()]
#   nodes = [onnx.helper.make_node("Identity", list(inps), list(inps), domain="")]
#   graph = onnx.helper.make_graph(nodes, f"test_{nm}", onnx_inputs, onnx_outputs)
#   return onnx.helper.make_model(graph, producer_name=f"test_{nm}")
#
# class TestOnnxBufferLoading(unittest.TestCase):
#   def test_const(self):
#     x = np.array(1.0, dtype=np.float32)
#     model = helper_make_identity_model({"X": x}, "identity")
#     runner = OnnxRunner(model)
#     # result = runner({"X": x})["X"]
#
# if __name__ == '__main__':
#   unittest.main()
#