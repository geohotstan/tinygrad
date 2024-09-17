from typing import List, Tuple
import functools, time, math
from wgpu.utils.device import get_default_device
from tinygrad.dtype import DType, PtrDType, dtypes, least_upper_dtype
from tinygrad.ops import UOp, UOps, TernaryOps, BinaryOps
from tinygrad.codegen.uopgraph import PatternMatcher, UPat
from tinygrad.device import Compiled, Allocator, Compiler
from tinygrad.renderer.cstyle import CStyleLanguage, ConstType
import wgpu

class WGSLCompiler(Compiler):
  def compile(self, src): return src.encode()

webgpu_matcher = PatternMatcher([
  # (UPat(UOps.ALU, name="x", dtype=dtypes.bool, arg=BinaryOps.MAX),
  #   lambda x: UOp(UOps.ALU, dtypes.float, tuple(s.cast(dtypes.float) for s in x.src), x.arg).cast(dtypes.bool)),
  # (UPat(UOps.LOAD, name="root", dtype=dtypes.bool, src=(UPat(name="x"),UPat(name="y"),UPat(name="z"),UPat(name="k"))),
  #   lambda root,x,y,z,k: UOp(root.op, dtypes.float, (x,y,z.cast(dtypes.float),k)).cast(dtypes.bool)),
  # (UPat(UOps.LOAD, name="root", dtype=dtypes.bool, src=(UPat(),UPat())),
  #   lambda root: UOp(root.op, dtypes.float, root.src, root.arg).cast(dtypes.bool)),
  # (UPat(UOps.STORE, name="root", src=(UPat(),UPat(),UPat(name="z",dtype=dtypes.bool), UPat())),
  #   lambda root,z: UOp(root.op, root.dtype, root.src[:2] + (z.cast(dtypes.float),), root.arg)),
  # (UPat(UOps.STORE, name="root", src=(UPat(),UPat(),UPat(name="z",dtype=dtypes.bool))),
  #   lambda root,z: UOp(root.op, root.dtype, root.src[:2] + (z.cast(dtypes.float),), root.arg)),
  # (UPat(UOps.STORE, name="root", src=(UPat(),UPat(),UPat(),UPat(name="g", dtype=dtypes.int))),
  #   lambda root,g: UOp(root.op, root.dtype, root.src[:3] + (g.cast(dtypes.float),), root.arg)),
  # (UPat(UOps.ALU, name="x", dtype=dtypes.bool, arg=BinaryOps.CMPNE, src=(UPat(),UPat(dtype=dtypes.bool))),
  #   lambda x: UOp(UOps.ALU, dtypes.bool, tuple(s.cast(dtypes.float) for s in x.src), x.arg).cast(x.dtype)),
  (UPat(UOps.ALU, name="x", dtype=dtypes.bool, arg=BinaryOps.MAX),
    lambda x: UOp(UOps.ALU, dtypes.float, tuple(s.cast(dtypes.float) for s in x.src), x.arg).cast(x.dtype)),
  # (UPat(UOps.ALU, name="x", dtype=dtypes.bool, arg=BinaryOps.MAX),
  #   lambda x: UOp(UOps.ALU, dtypes.float, tuple(s.cast(dtypes.float) for s in x.src), x.arg).cast(x.dtype)),
  # (UPat(UOps.ALU, name="x", dtype=dtypes.bool, arg=BinaryOps.CMPNE, custom_early_reject=set([(UOps.ALU, None)])),
  #   lambda x: UOp(UOps.ALU, dtypes.bool, x.src, x.arg).cast(dtypes.float)),

  # (UPat(UOps.DEFINE_ACC, name="x", dtype=dtypes.bool),
  #  lambda x: UOp(UOps.DEFINE_ACC, dtypes.float, tuple(s.cast(dtypes.float) for s in x.src), x.arg)),
  # (UPat.var("x", dtypes.bool).ne(UPat.var("y")),
  #   lambda x, y: ),
  # (UPat(UOps.STORE, name="x", dtype=dtypes.void),
  #   lambda x: UOp(UOps.STORE, x.dtype, cast_same_dtype(x.src), x.arg))
  # (UPat(UOps.STORE, dtypes.void, name="x"), lambda x: UOp(UOps.STORE, x.dtype, (s.cast(dtypes.float) for s in x.src), x.arg))

  # (UPat(UOps.DEFINE_GLOBAL, name="x", dtype=PtrDType(dtypes.bool)),
  #   lambda x: UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), x.src, x.arg)),
  # (UPat((UOps.CONST, UOps.DEFINE_ACC), name="x", dtype=dtypes.bool),
  #   lambda x: UOp(x.op, dtypes.float, x.src, float(x.arg) if isinstance(x.arg, bool) else x.arg)),
  # (UPat(UOps.STORE, name="root", dtype=dtypes.void, src=(UPat(UOps.DEFINE_GLOBAL, name="x", dtype=PtrDType(dtypes.bool)),
  #                                                        UPat(name="y"),UPat(UOps.ALU, name="alu", dtype=dtypes.bool))),
  #   lambda root, x, y, alu: UOp(root.op, root.dtype, (UOp(x.op, dtype=PtrDType(dtypes.float), arg=x.arg), y, alu.cast(dtypes.float)), root.arg)),
  # (UPat(UOps.STORE, name="root", dtype=dtypes.void, src=(UPat(UOps.DEFINE_GLOBAL, name="x", dtype=PtrDType(dtypes.bool)),
  #                                                        UPat(name="y"),UPat(name="z"))),
  #   lambda root, x, y, z: UOp(root.op, root.dtype, (UOp(x.op, dtype=PtrDType(dtypes.float), arg=x.arg), y, alu.cast(dtypes.float)), root.arg)),
  # (UPat(UOps.ALU, name="x", dtype=dtypes.bool, arg=BinaryOps.MAX),
  #   lambda x: UOp(UOps.ALU, *cast_same_dtype(x.src), x.arg).cast(x.dtype)),
    # (UPat(UOps.ALU, name="x", dtype=dtypes.bool, arg=BinaryOps.CMPNE),
    # lambda x: UOp(UOps.ALU, dtypes.bool, tuple(s.cast(dtypes.float) for s in x.src), x.arg).cast(dtypes.bool)),
])

class WGSLRenderer(CStyleLanguage):
  code_for_workitem = {"g": lambda x: f"i32(gindex.{'xyz'[int(x)]})", "l": lambda x: f"i32(lindex.{'xyz'[int(x)]})"}
  supports_float4 = False
  type_map = {dtypes.long: "i64", dtypes.float: "f32", dtypes.half: "f16", dtypes.int32: "i32", dtypes.uint32: "u32", dtypes.bool: "bool",
              dtypes.uint8: "u32", dtypes.uint16: "u32", dtypes.ulong: "u64"}
  barrier: str = "workgroupBarrier();"
  infinity: str = "inf(1.0)"
  nan: str = "nan()"
  extra_matcher = webgpu_matcher
  code_for_op = { **CStyleLanguage().code_for_op,
                 TernaryOps.MULACC: lambda x,y,z,dtype: f"fma({x},{y},{z})",
                 TernaryOps.WHERE: lambda a,b,c,dtype: f"select({c},{b},{a})" }

  def render_const(self, x:ConstType, dtype:DType) -> str:
    if math.isnan(x): val = self.nan
    elif math.isinf(x): val = ("-" if x < 0 else "") + self.infinity
    elif dtype == dtypes.bool: val = "true" if x else "false"
    else: val = str(x)
    return (self.render_cast(val, dtype) if dtype not in [dtypes.float, dtypes.int, dtypes.bool] else val)

  def render_cast(self, x:str, var_dtype:DType, bitcast=False) -> str:
    if self.type_map[var_dtype]: return f"bitcast<{self.type_map[var_dtype]}>({x})" if bitcast else f"{self.type_map[var_dtype]}({x})"
    raise NotImplementedError(f"no cast for {var_dtype}")
  def render_dtype(self, dtype): return "var"
  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool]]], uops:List[UOp], prefix=None) -> str:
    # HACK =====
    # pull local workgroup into external prekernel
    # no stateful impl of renderer
    prekernel = [k for k in kernel if "<workgroup>" in k]
    kernel = [k for k in kernel if "<workgroup>" not in k]
    # ==========

    local_size = [u.arg[1] for u in uops if u.op is UOps.SPECIAL and u.arg[0][0] == 'l']
    if not local_size: local_size = [1]
    bind_it = iter(range(len(bufs)))
    prg = "fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }\nfn inf(a: f32) -> f32 { return a/0.0; }\n"
    prg += "\n".join(prekernel+[f"@group(0) @binding({next(bind_it)}) {'var<storage,read_write>' if isinstance(dtype, PtrDType) else 'var<uniform>'} {name}: {f'array<{self.type_map[dtype]}>' if isinstance(dtype, PtrDType) else 'i32'};" for name,(dtype,rw) in bufs])  # noqa: E501
    prg += f"\n@compute @workgroup_size({','.join([str(x) for x in local_size])}) fn {function_name}(@builtin(workgroup_id) gindex: vec3<u32>, @builtin(local_invocation_id) lindex: vec3<u32>) {{\n" + "\n".join(kernel) + "\n}"  # noqa: E501
    return prg
  def render_local(self, name: str, dtype:DType, size: int): return f"var<workgroup> {name}: array<{self.type_map[dtype]}, {size}>;"

def create_uniform(wgpu_device, val: int) -> wgpu.GPUBuffer:
  buf = wgpu_device.create_buffer(size=4, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
  wgpu_device.queue.write_buffer(buf, 0, val.to_bytes(4, "little"))
  return buf

class WebGPUProgram:
  def __init__(self, device, name:str, lib:bytes):
    self.device = device
    self.name, self.lib, self.prg = name, lib, self.device.create_shader_module(code=lib.decode())   # NOTE: this is the compiler
  def __call__(self, *bufs, global_size, local_size, vals=(), wait=False):
    binding_layouts = [{"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform if i >= len(bufs) else wgpu.BufferBindingType.storage }} for i in range(len(bufs)+len(vals))]  # noqa: E501
    bindings = [{"binding": i,
                 "resource": {"buffer": create_uniform(self.device, x) if i >= len(bufs) else x,
                              "offset": 0,
                              "size": 4 if i >= len(bufs) or x.size < 4 else x.size}}
                              for i,x in enumerate(bufs+vals)]  # noqa: E501
    from tinygrad.helpers import DEBUG
    if DEBUG == 4:
      print(bindings)
      print("fuck")
      print("fuck")
      print("fuck")
    bind_group_layout = self.device.create_bind_group_layout(entries=binding_layouts)
    pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
    bind_group = self.device.create_bind_group(layout=bind_group_layout, entries=bindings)
    compute_pipeline = self.device.create_compute_pipeline(layout=pipeline_layout,compute={"module": self.prg, "entry_point": self.name},)
    command_encoder = self.device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group, [], 0, 999999) # last 2 not used
    compute_pass.dispatch_workgroups(*global_size)  # x y z
    compute_pass.end()
    st = time.perf_counter()
    self.device.queue.submit([command_encoder.finish()])
    return time.perf_counter() - st

class WebGpuAllocator(Allocator):
  def __init__(self, device): self.device = device
  def _alloc(self, size: int, options):
    return self.device.create_buffer(size=size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
  def copyin(self, dest: wgpu.GPUBuffer, src: memoryview): self.device.queue.write_buffer(dest, 0, src)
  def copyout(self, dest:memoryview, src: wgpu.GPUBuffer): dest[:] = self.device.queue.read_buffer(src, 0)    # TODO: remove this copy

class WebGpuDevice(Compiled):
  def __init__(self, device:str):
    wgpu_device = get_default_device()
    super().__init__(device, WebGpuAllocator(wgpu_device), WGSLRenderer(), WGSLCompiler(), functools.partial(WebGPUProgram, wgpu_device))
                     #CompilerOptions(device="WEBGPU", supports_float4=False, local_max=[256, 256, 64],
                     #                                     global_max=[65535, 65535, 65535]), WGSLRenderer, lambda x: x, WebGPUProgram)
