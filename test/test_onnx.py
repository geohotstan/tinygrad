import unittest, os
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import CI
from extra.onnx import OnnxRunner, OnnxValue, OnnxNode
import onnx
import tempfile

def create_identity_model(inputs, outputs):
  nodes = [onnx.helper.make_node("Identity", inputs=[i.name for i in inputs], outputs=[o.name for o in outputs])]
  graph = onnx.helper.make_graph(name="test_graph", inputs=inputs, outputs=outputs, nodes=nodes)
  return onnx.helper.make_model(graph)

class TestOnnxSimple(unittest.TestCase):
  def setUp(self):
    inputs = [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2])]
    outputs = [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2])]
    self.model = create_identity_model(inputs, outputs)

  def _verify_load(self, runner:OnnxRunner):
    self.assertEqual(len(runner.graph_nodes), 1)
    self.assertEqual(runner.graph_nodes[0].op, "Identity")
    self.assertEqual(len(runner.graph_inputs), 1)
    self.assertEqual(len(runner.graph_outputs), 1)
    self.assertEqual(runner.graph_inputs["x"], OnnxValue((1,2), dtypes.float, False, False))
    self.assertEqual(runner.graph_outputs["y"], OnnxValue((1,2), dtypes.float, False, False))

  def test_load(self):
    # test load from bytes
    runner = OnnxRunner(self.model.SerializeToString())
    self._verify_load(runner)

    # test load from file path
    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmpfile:
      onnx.save(self.model, tmpfile.name)
      runner = OnnxRunner(tmpfile.name)
      self._verify_load(runner)

  def test_runner(self):
    runner = OnnxRunner(self.model.SerializeToString())
    input_data = {"x": np.array([[1.0, 2.0]], dtype=np.float32)}
    output = runner(input_data)
    np.testing.assert_array_equal(output["y"].numpy(), input_data["x"])

  def test_jit_runner(self):
    runner = OnnxRunner(self.model.SerializeToString(), jit=True)
    input_data = {"x": np.array([[1.0, 2.0]], dtype=np.float32)}
    output = runner(input_data)
    np.testing.assert_array_equal(output["y"].numpy(), input_data["x"])

  # def test_model_serialization(self):
  #   ser = self.model.SerializeToString()
  #   onnx.save(self.model, )


class TestOnnxInputs(unittest.TestCase):
  def test_variable_dimensions(self):
    inputs = [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["h", "w"])]
    outputs = [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, ["w", "h"])]
    nodes = [onnx.helper.make_node("Transpose", inputs=["x"], outputs=["y"], perm=[1,0])]
    graph = onnx.helper.make_graph(name="test_graph", inputs=inputs, outputs=outputs, nodes=nodes)
    model = onnx.helper.make_model(graph)
    runner = OnnxRunner(model)
    input_data = {"x": np.array([[1.0, 2.0]], dtype=np.float32)}
    runner(input_data)

  def test_optional_type(self):
    tensor_type = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [1, 2])
    optional_type = onnx.helper.make_optional_type_proto(tensor_type)
    value_info = onnx.helper.make_value_info("x", optional_type)
    inputs = [value_info]
    outputs = [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2])]
    model = create_identity_model(inputs, outputs)
    runner = OnnxRunner(model.SerializeToString())
    self.assertTrue(runner.graph_inputs["x"].is_optional)

    # test real input
    input_data = {"x": np.array([[1.0, 2.0]], dtype=np.float32)}
    output = runner(input_data)
    np.testing.assert_array_equal(output["y"].numpy(), input_data["x"])
    input_data = {"x": Tensor([[1.0, 2.0]], dtype=dtypes.float32)}
    output = runner(input_data)
    np.testing.assert_array_equal(output["y"].numpy(), input_data["x"].numpy())

    # test optional input
    input_data = {"x": None}
    output = runner(input_data)
    np.testing.assert_array_equal(output["y"], input_data["x"])

  def test_sequence_type(self):
    tensor_type = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [1, 2])
    sequence_type = onnx.helper.make_sequence_type_proto(tensor_type)
    value_info = onnx.helper.make_value_info("x", sequence_type)
    inputs = [value_info]
    outputs = [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2])]
    model = create_identity_model(inputs, outputs)
    runner = OnnxRunner(model.SerializeToString())
    self.assertTrue(runner.graph_inputs["x"].is_sequence)

    # test real input
    input_data = {"x": [np.array([[1.0, 2.0]], dtype=np.float32), np.array([[3.0, 4.0]], dtype=np.float32)]}
    output = runner(input_data)
    for y, x in zip(output["y"], input_data["x"]): np.testing.assert_array_equal(y.numpy(), x)
    input_data = {"x": [Tensor([[1.0, 2.0]], dtype=dtypes.float32), Tensor([[3.0, 4.0]], dtype=dtypes.float32)]}
    output = runner(input_data)
    for y, x in zip(output["y"], input_data["x"]): np.testing.assert_array_equal(y.numpy(), x.numpy())

    # test empty sequence input
    input_data = {"x": []}
    output = runner(input_data)
    self.assertEqual(output["y"], [])




class TestOnnxTraining(unittest.TestCase):
  ...


  def test_sequence_type(self):
    ...
  # def test_errors(self):
  #   runner = OnnxRunner(self.model_path)

  #   # wrong shape
  #   with self.assertRaises(RuntimeError):
  #     input_data = {"x": np.array([1.0, 2.0], dtype=np.float32)}
  #     runner(input_data)

  #   # wrong dtype
  #   with self.assertRaises(RuntimeError):
  #     input_data = {"x": np.array([[1, 2]], dtype=np.int32)}
  #     runner(input_data)

  #   #

@unittest.skipIf(not CI, "requires internet access")
class TestOnnxMnist(unittest.TestCase):
  def setUp(self):
    url = "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mnist/model/mnist-1.onnx"
    self.runner = OnnxRunner(url)

  def test_load(self):
    self.assertEqual(len(self.runner.graph_nodes), 24)

  def test_run(self):
    input_data = {"Input3": np.random.rand(1, 1, 28, 28).astype(np.float32)}
    output = self.runner(input_data)
    self.assertIn("Plus214_Output_0", output)
    self.assertEqual(output["Plus214_Output_0"].shape, (1, 10))

  def test_run_jit(self):
    runner = OnnxRunner("https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mnist/model/mnist-1.onnx", jit=True)
    input_data = {"Input3": np.random.rand(1, 1, 28, 28).astype(np.float32)}
    output = runner(input_data)
    self.assertIn("Plus214_Output_0", output)
    self.assertEqual(output["Plus214_Output_0"].shape, (1, 10))

if __name__ == "__main__":
  unittest.main()