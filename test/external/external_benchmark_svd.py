import ast, argparse, statistics, subprocess, textwrap, time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

from tinygrad import Tensor, Device, dtypes, GlobalCounters
from tinygrad.device import Compiler
from tinygrad.engine.realize import method_cache


@dataclass
class CompileStats:
  calls: int = 0
  wall_s: float = 0.0


@dataclass
class CaseMetrics:
  shape: tuple[int, int]
  full_matrices: bool
  variant: str
  cold_compile_calls: int
  cold_compile_wall_s: float
  cold_kernel_wall_s: float
  cold_kernel_sum_s: float
  cold_kernel_count: int
  cold_total_wall_s: float
  steady_compile_calls_median: float
  steady_compile_wall_s_median: float
  steady_kernel_wall_s_median: float
  steady_kernel_sum_s_median: float
  steady_kernel_count_median: float


def parse_shapes(spec: str) -> list[tuple[int, int]]:
  out: list[tuple[int, int]] = []
  for tok in spec.split(","):
    tok = tok.strip().lower()
    if not tok: continue
    m, n = tok.split("x")
    out.append((int(m), int(n)))
  if len(out) == 0: raise ValueError("no shapes parsed")
  return out


def _extract_ref_svd(ref: str) -> Callable:
  src = subprocess.check_output(["git", "show", f"{ref}:tinygrad/tensor.py"], text=True)
  tree = ast.parse(src)
  fn_src = None
  for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == "Tensor":
      for sub in node.body:
        if isinstance(sub, ast.FunctionDef) and sub.name == "svd":
          fn_src = ast.get_source_segment(src, sub)
          break
      break
  if fn_src is None: raise RuntimeError(f"could not find Tensor.svd in ref {ref}")

  import tinygrad.tensor as tensor_mod
  glb = dict(tensor_mod.__dict__)
  loc: dict[str, Callable] = {}
  exec(textwrap.dedent(fn_src), glb, loc)
  return loc["svd"]


@contextmanager
def patch_svd(svd_impl: Callable):
  original = Tensor.svd
  Tensor.svd = svd_impl
  try:
    yield
  finally:
    Tensor.svd = original


@contextmanager
def capture_compile_time(force_recompile: bool):
  original = Compiler.compile_cached
  stats = CompileStats()

  def wrapped(self, src: str):
    st = time.perf_counter()
    ret = type(self).compile(self, src) if force_recompile else original(self, src)
    stats.calls += 1
    stats.wall_s += time.perf_counter() - st
    return ret

  Compiler.compile_cached = wrapped
  try:
    yield stats
  finally:
    Compiler.compile_cached = original


def build_schedule(inp: Tensor, full_matrices: bool):
  U, S, Vh = inp.svd(full_matrices=full_matrices)
  return U.schedule_with_vars(S, Vh)


def run_schedule_wait(schedule, var_vals: dict[str, int]) -> float:
  total = 0.0
  for ei in schedule:
    et = ei.run(var_vals, wait=True, do_update_stats=True)
    if et is not None: total += et
  return total


def run_case(shape: tuple[int, int], full_matrices: bool, repeats: int, variant: str, force_recompile: bool) -> CaseMetrics:
  Device[Device.DEFAULT].synchronize()
  inp = Tensor.randn(*shape, dtype=dtypes.float).realize()
  Device[inp.device].synchronize()

  method_cache.clear()
  GlobalCounters.reset()
  schedule, var_vals = build_schedule(inp, full_matrices=full_matrices)
  with capture_compile_time(force_recompile) as compile_stats:
    st_total = time.perf_counter()
    for ei in schedule: ei.lower()
    kernel_sum = run_schedule_wait(schedule, var_vals)
    cold_total_wall = time.perf_counter() - st_total
  cold_kernel_wall = GlobalCounters.time_sum_s
  cold_kernel_count = GlobalCounters.kernel_count

  steady_compile_calls: list[int] = []
  steady_compile_wall: list[float] = []
  steady_kernel_wall: list[float] = []
  steady_kernel_sum: list[float] = []
  steady_kernel_count: list[int] = []
  for _ in range(repeats):
    GlobalCounters.reset()
    schedule, var_vals = build_schedule(inp, full_matrices=full_matrices)
    with capture_compile_time(force_recompile=False) as cstats:
      ksum = run_schedule_wait(schedule, var_vals)
    steady_compile_calls.append(cstats.calls)
    steady_compile_wall.append(cstats.wall_s)
    steady_kernel_wall.append(GlobalCounters.time_sum_s)
    steady_kernel_sum.append(ksum)
    steady_kernel_count.append(GlobalCounters.kernel_count)

  return CaseMetrics(
    shape=shape, full_matrices=full_matrices, variant=variant,
    cold_compile_calls=compile_stats.calls,
    cold_compile_wall_s=compile_stats.wall_s,
    cold_kernel_wall_s=cold_kernel_wall,
    cold_kernel_sum_s=kernel_sum,
    cold_kernel_count=cold_kernel_count,
    cold_total_wall_s=cold_total_wall,
    steady_compile_calls_median=statistics.median(steady_compile_calls),
    steady_compile_wall_s_median=statistics.median(steady_compile_wall),
    steady_kernel_wall_s_median=statistics.median(steady_kernel_wall),
    steady_kernel_sum_s_median=statistics.median(steady_kernel_sum),
    steady_kernel_count_median=statistics.median(steady_kernel_count),
  )


def format_row(x: CaseMetrics) -> str:
  shp = f"{x.shape[0]}x{x.shape[1]}"
  fm = "full" if x.full_matrices else "thin"
  return (
    f"{x.variant:>8} {shp:>8} {fm:>4} | "
    f"cold compile {x.cold_compile_wall_s*1e3:8.2f} ms ({x.cold_compile_calls:3d} calls) | "
    f"cold kernel {x.cold_kernel_wall_s*1e3:8.2f} ms | "
    f"steady kernel {x.steady_kernel_wall_s_median*1e3:8.2f} ms | "
    f"steady compile {x.steady_compile_wall_s_median*1e3:8.3f} ms"
  )


def print_deltas(curr: list[CaseMetrics], base: list[CaseMetrics]) -> None:
  base_map = {(x.shape, x.full_matrices): x for x in base}
  print("\nDeltas (current vs baseline):")
  for x in curr:
    b = base_map[(x.shape, x.full_matrices)]
    cc = (x.cold_compile_wall_s / b.cold_compile_wall_s - 1.0) * 100 if b.cold_compile_wall_s > 0 else 0.0
    ck = (x.cold_kernel_wall_s / b.cold_kernel_wall_s - 1.0) * 100 if b.cold_kernel_wall_s > 0 else 0.0
    sk = (x.steady_kernel_wall_s_median / b.steady_kernel_wall_s_median - 1.0) * 100 if b.steady_kernel_wall_s_median > 0 else 0.0
    shp = f"{x.shape[0]}x{x.shape[1]}"
    fm = "full" if x.full_matrices else "thin"
    print(f"{shp:>8} {fm:>4} | cold compile {cc:+7.2f}% | cold kernel {ck:+7.2f}% | steady kernel {sk:+7.2f}%")


def main():
  parser = argparse.ArgumentParser(description="Benchmark Tensor.svd compile and kernel times")
  parser.add_argument("--shapes", type=str, default="64x64,128x128,256x256,256x128,128x256")
  parser.add_argument("--full-matrices", choices=["both", "full", "thin"], default="both")
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--baseline-ref", type=str, default="HEAD", help="git ref used as baseline Tensor.svd")
  parser.add_argument("--no-baseline", action="store_true", help="only benchmark current working tree Tensor.svd")
  parser.add_argument("--force-recompile", action="store_true", help="measure cold compile by forcing compile (ignore compiler disk cache)")
  args = parser.parse_args()

  shapes = parse_shapes(args.shapes)
  modes = [True, False] if args.full_matrices == "both" else [args.full_matrices == "full"]
  print(f"device={Device.DEFAULT}, repeats={args.repeats}, shapes={shapes}, modes={['full' if x else 'thin' for x in modes]}")

  current_impl = Tensor.svd
  baseline_impl = None if args.no_baseline else _extract_ref_svd(args.baseline_ref)

  baseline_results: list[CaseMetrics] = []
  if baseline_impl is not None:
    with patch_svd(baseline_impl):
      for shape in shapes:
        for full_matrices in modes:
          baseline_results.append(run_case(shape, full_matrices, args.repeats, "baseline", force_recompile=args.force_recompile))

  current_results: list[CaseMetrics] = []
  with patch_svd(current_impl):
    for shape in shapes:
      for full_matrices in modes:
        current_results.append(run_case(shape, full_matrices, args.repeats, "current", force_recompile=args.force_recompile))

  print("\nResults:")
  for row in baseline_results + current_results:
    print(format_row(row))

  if baseline_results:
    print_deltas(current_results, baseline_results)


if __name__ == "__main__":
  main()
