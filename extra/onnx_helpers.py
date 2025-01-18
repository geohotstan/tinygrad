from tinygrad import Tensor, Device
from extra.onnx import OnnxRunner, OnnxValue
import numpy as np
import onnxruntime as ort

def get_example_input(spec:OnnxValue) -> Tensor:
  assert not spec.is_optional and not spec.is_sequence, "only allow tensor input for now"
  # assign naive shape for variable dimension for now
  shape = tuple(dim if isinstance(dim, int) else 1 for dim in spec.shape)
  # assign naive values for now
  # TODO: what is this mul(8) for?
  # return Tensor.randn(*shape, dtype=spec.dtype, device=Device.DEFAULT).mul(8).realize()
  return Tensor.randn(*shape, dtype=spec.dtype, device=Device.DEFAULT).realize()

def validate(fn, rtol=1e-5, atol=1e-5):
  run_onnx = OnnxRunner(fn)
  new_inputs = {name:get_example_input(spec) for name, spec in run_onnx.graph_inputs.items()}
  tinygrad_out = run_onnx(new_inputs)

  ort_sess = ort.InferenceSession(fn)
  np_inputs = {k:v.numpy() for k,v in new_inputs.items()}
  out_names = list(run_onnx.graph_outputs)
  out_values = ort_sess.run(out_names, np_inputs)
  ort_out = dict(zip(out_names, out_values))

  assert len(tinygrad_out) == len(ort_out) and tinygrad_out.keys() == ort_out.keys()
  for k in tinygrad_out.keys():
    tiny_v, onnx_v = tinygrad_out[k], ort_out[k]
    if tiny_v is None: assert tiny_v == onnx_v
    else: np.testing.assert_allclose(tiny_v.numpy(), onnx_v, rtol=rtol, atol=atol, err_msg=f"For tensor '{k}' in {tinygrad_out.keys()}")