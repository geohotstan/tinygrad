import sys, onnx, time
from tinygrad import Tensor, GlobalCounters, fetch
from extra.onnx import OnnxRunner, OnnxValue

def get_example_input(spec:OnnxValue) -> Tensor:
  assert not spec.is_optional and not spec.is_sequence, "only allow tensor input for now"
  # assign naive shape for variable dimension for now
  shape = tuple(dim if isinstance(dim, int) else 1 for dim in spec.shape)
  # assign naive values for now
  return Tensor.randn(*shape, dtype=spec.dtype).mul(8).realize()

def load_onnx_model(fn):
  onnx_file = fetch(fn)
  onnx_model = onnx.load(onnx_file)
  return OnnxRunner(onnx_model)

if __name__ == "__main__":
  run_onnx = load_onnx_model(sys.argv[1])
  print("loaded model")

  for i in range(3):
    new_inputs = {name:get_example_input(spec) for name, spec in run_onnx.graph_inputs.items()}
    GlobalCounters.reset()
    print(f"run {i}")
    run_onnx.jit_runner(**new_inputs)

  # run 20 times
  for _ in range(20):
    new_inputs = {name:get_example_input(spec) for name, spec in run_onnx.graph_inputs.items()}
    GlobalCounters.reset()
    st = time.perf_counter()
    out = run_onnx.jit_runner(**new_inputs)
    mt = time.perf_counter()
    val = next(iter(out.values())).numpy()
    et = time.perf_counter()
    print(f"enqueue {(mt-st)*1e3:6.2f} ms -- total run {(et-st)*1e3:6.2f} ms")
