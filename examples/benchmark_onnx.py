import sys, onnx, time
from tinygrad import Tensor, TinyJit, GlobalCounters, fetch, Device, getenv
from extra.onnx import OnnxRunner
from extra.onnx_helpers import get_example_input, validate

if __name__ == "__main__":
  onnx_file = fetch(sys.argv[1])
  run_onnx = OnnxRunner(onnx_file)
  print("loaded model")

  run_onnx_jit = TinyJit(lambda **kwargs: next(iter(run_onnx({k:v.to(Device.DEFAULT) for k,v in kwargs.items()}).values())), prune=True)
  for i in range(3):
    new_inputs = {name:get_example_input(spec) for name, spec in run_onnx.graph_inputs.items()}
    GlobalCounters.reset()
    print(f"run {i}")
    run_onnx_jit(**new_inputs)

  # run 20 times
  for _ in range(20):
    new_inputs = {name:get_example_input(spec) for name, spec in run_onnx.graph_inputs.items()}
    GlobalCounters.reset()
    st = time.perf_counter()
    out = run_onnx_jit(**new_inputs)
    mt = time.perf_counter()
    val = out.numpy()
    et = time.perf_counter()
    print(f"enqueue {(mt-st)*1e3:6.2f} ms -- total run {(et-st)*1e3:6.2f} ms")

  if getenv("ORT"):
    validate(onnx_file, rtol=1e-3, atol=1e-3)
    print("model validated")