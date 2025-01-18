import sys, onnx, time
from tinygrad import Tensor, GlobalCounters, fetch, Device
from extra.onnx import OnnxRunner, OnnxValue
import numpy as np
import onnxruntime as ort

def get_example_input(spec:OnnxValue) -> Tensor:
  assert not spec.is_optional and not spec.is_sequence, "only allow tensor input for now"
  # assign naive shape for variable dimension for now
  shape = tuple(dim if isinstance(dim, int) else 1 for dim in spec.shape)
  # assign naive values for now
  return Tensor.randn(*shape, dtype=spec.dtype, device=Device.DEFAULT).mul(8).realize()

if __name__ == "__main__":
  onnx_file = fetch(sys.argv[1])
  onnx_model = onnx.load(onnx_file)
  run_onnx = OnnxRunner(onnx_model, jit=True)
  print("loaded model")

  for i in range(3):
    new_inputs = {name:get_example_input(spec) for name, spec in run_onnx.graph_inputs.items()}
    GlobalCounters.reset()
    print(f"run {i}")
    run_onnx(new_inputs)

  # run 20 times
  for _ in range(20):
    new_inputs = {name:get_example_input(spec) for name, spec in run_onnx.graph_inputs.items()}
    GlobalCounters.reset()
    st = time.perf_counter()
    out = run_onnx(new_inputs)
    mt = time.perf_counter()
    val = next(iter(out.values())).numpy()
    et = time.perf_counter()
    print(f"enqueue {(mt-st)*1e3:6.2f} ms -- total run {(et-st)*1e3:6.2f} ms")

  tiny_out = [v.numpy() for v in out.values()]
  sess = ort.InferenceSession(onnx_file)
  ort_out = sess.run([out.name for out in onnx_model.graph.output], {k:v.numpy() for k,v in new_inputs.items()})
  rtol, atol = 1e-3, 1e-3
  assert len(tiny_out) == len(ort_out)
  for tiny_v, ort_v in zip(tiny_out, ort_out):
    if tiny_v is None: assert tiny_v == ort_v
    else: np.testing.assert_allclose(tiny_v, ort_v, rtol=rtol, atol=atol)
  del sess
  print("ort test passed")
