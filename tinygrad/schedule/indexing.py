from typing import Iterator
import functools, itertools
from dataclasses import dataclass, field, replace
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp, graph_rewrite, sint, AxisType, rewrite_group, broadcast_axes
from tinygrad.uop.ops import gate_kernel_sink, _broadcast_shape
from tinygrad.uop.symbolic import symbolic, pm_simplify_valid, pm_drop_and_clauses
from tinygrad.helpers import argsort, all_same, cpu_profile, PCONTIG, colored, Context, SPEC

@dataclass
class IndexingContext:
  realize_map: dict[UOp, None|list[int]] = field(default_factory=dict)
  non_removable: dict[UOp, None] = field(default_factory=dict)
  range_map: dict[UOp, tuple[tuple[UOp, ...], tuple[UOp, ...]]] = field(default_factory=dict)
  # view-chain nodes of a tensor-coordinate gather: the buffer side is left unindexed by apply
  # rangeify and folded into the flat read by lower_shaped_gather instead
  gather_views: set[UOp] = field(default_factory=set)
  # shaped-gather tag -> its recorded output ranges, for the emission pass
  shaped_kept: dict[str, tuple[UOp, ...]] = field(default_factory=dict)
  gather_tag: Iterator[int] = field(default_factory=itertools.count)

  # create ranges
  range_idx: Iterator[int] = field(default_factory=itertools.count)
  def new_range(self, s:sint, axistype:AxisType=AxisType.WEAK) -> UOp:
    if isinstance(s, UOp) and s.op is Ops.RANGE: return s
    # if a range has a 1 src, it's the same as UOp.const(0)
    return UOp.range(s, next(self.range_idx), axistype) if resolve(s!=1) else UOp.const(0)


ALWAYS_CONTIGUOUS: set[Ops] = {Ops.CONTIGUOUS, Ops.AFTER, Ops.BUFFER,
                      Ops.CONST, Ops.MSELECT, Ops.MSTACK, Ops.PARAM,
                      Ops.LOAD, Ops.CALL, Ops.FUNCTION}

def realize(ctx:IndexingContext, tr:UOp) -> None: ctx.realize_map[tr] = None

def realize_srcs(ctx:IndexingContext, rb:UOp) -> None:
  for s in rb.src:
    if s.base.op not in ALWAYS_CONTIGUOUS: ctx.realize_map[s] = None

def realize_store_after_src(ctx:IndexingContext, dest:UOp, src:UOp):
  # you don't usually have to do this for assign unless there's a WAR hazard like TestAssign.test_assign_double_diamond_reduce
  if dest.base in src.toposort(enter_calls=False): ctx.realize_map[src] = None

def realize_custom_kernel_srcs(ctx:IndexingContext, c:UOp) -> None:
  for s in c.src[1:]:
    while s.op is Ops.RESHAPE: s = s.src[0]
    if s.op not in ALWAYS_CONTIGUOUS:
      ctx.realize_map[s] = None
      ctx.non_removable[s] = None

pm_generate_realize_map = PatternMatcher([
  # realize the inputs of custom kernel calls
  (UPat(Ops.CALL, src=(UPat((Ops.SINK, Ops.PROGRAM)),), name="c", allow_any_len=True), realize_custom_kernel_srcs),
  # always realize
  (UPat({Ops.CONTIGUOUS, Ops.STORE}, name="tr"), realize),
  # realize srcs of these
  (UPat((Ops.MSELECT, Ops.MSTACK), name="rb"), realize_srcs),
  # sometimes we need to realize the src of STORE if there's a self-access
  (UPat(Ops.STORE, src=(UPat.var("dest"), UPat.var("src"))), realize_store_after_src),
])

@dataclass(frozen=True)
class BufferizeOpts:
  # on AddrSpace.LOCAL, device is the id
  device: str|tuple[str, ...]|int|None
  addrspace: AddrSpace = AddrSpace.GLOBAL
  removable: bool = True

def broadcast_rngs(x:UOp, src:UOp, rngs:tuple[UOp, ...]) -> tuple[UOp, ...]:
  # NOTE: a gather INDEX is not Broadcastable, so a node on its buffer side inherits its rngs
  # verbatim; tensor coordinates never appear in rng tuples, they are data flow (see the
  # shaped-INDEX branch in run_rangeify and lower_shaped_gather)
  if x.op not in GroupOp.Broadcastable: return rngs
  baxes, nleft = broadcast_axes(src.shape, x.shape), len(x.shape)-len(src.shape)
  return tuple(r.const_like(0) if j in baxes else r for j,r in enumerate(rngs) if j >= nleft)

# TODO: srcs contain (real data srcs, something else, ranges) and the boundary is confusing. see range_start
def data_srcs(op:Ops, src:tuple[UOp, ...]) -> tuple[UOp, ...]:
  if op in {Ops.PARAM, Ops.BUFFER, Ops.RANGE, Ops.SPECIAL}: return ()
  # the store of a bound Variable only carries the input value, it has no data srcs
  if op is Ops.STORE and src[0].is_variable: return ()
  if op is Ops.INDEX:
    # the shaped index srcs of a gather are data flow, they compute the address
    return src[:1] + tuple(s for s in src[1:] if s.shape != ())
  if op in GroupOp.Movement|{Ops.STAGE, Ops.REDUCE, Ops.AFTER, Ops.END}: return src[:1]
  return src

def create_bufferize_and_index_srcs(ctx:IndexingContext, x:UOp) -> list[UOp]:
  new_srcs = []
  # shape/bound/index args that are not data src should not be indexed
  data_src_count = len(data_srcs(x.op, x.src))
  # a gather with tensor coordinates: the buffer side is read by lower_shaped_gather, which
  # folds the view chain itself, so nothing on it is pre-indexed here (its rng bookkeeping
  # carries placeholder ranges, not reads). the coords are data flow, indexed by the coordinate
  # (big) ranges they are consumed at, never by the gather's buffer-frame rngs
  shaped_gather = x.op is Ops.INDEX and len(x.src) > 1 and any(s.shape != () for s in x.src[1:])
  for i, s in enumerate(x.src):
    new_src = s
    is_coord = shaped_gather and i > 0 and s.shape != () and x in ctx.range_map
    skip = (not is_coord) and (x in ctx.gather_views or shaped_gather)
    src_rngs = broadcast_rngs(x, s, ctx.range_map[x][0]) if x in ctx.range_map else ()
    if is_coord:
      big_shape = _broadcast_shape(*(t.shape for t in x.src[1:]))
      baxes, nleft = broadcast_axes(s.shape, big_shape), len(big_shape)-len(s.shape)
      out_rngs = ctx.range_map[x][1]
      src_rngs = tuple(r.const_like(0) if j in baxes else r for j,r in enumerate(out_rngs[:len(big_shape)]) if j >= nleft)
    if s.op in {Ops.PARAM, Ops.BUFFER, Ops.MSTACK, Ops.MSELECT, Ops.AFTER}:
      # NOTE: for a shaped INDEX, src_rngs can be shorter than the buffer shape when the movement
      # chain below already folded the index srcs into a single flat index. don't index again
      if x in ctx.range_map and i < data_src_count and not skip and len(src_rngs) == len(s.shape):
        new_src = new_src.index(*src_rngs)
    elif s in ctx.realize_map and (i < data_src_count or s.op is Ops.STORE):
      import os
      if os.environ.get('TRACE_GATE'): print('STAGEING:', s.op.name, 'consumer:', x.op.name)
      # NOTE: the i gate keeps shape/bound args out (a realized shape arg must never be staged),
      # STORE srcs of AFTER still take their ENDs
      realized_ranges = ctx.realize_map[s]
      assert isinstance(realized_ranges, list), "realize map must contain range list"
      closed_ranges = tuple([r for i,r in enumerate(ctx.range_map[s][1]) if i in realized_ranges])
      if s.op is Ops.STORE:
        # add the ends if this is a store
        new_src = s.end(*[r for r in closed_ranges if r.op is Ops.RANGE])
        del ctx.realize_map[s]
      else:
        removable = s.op not in ALWAYS_CONTIGUOUS and s not in ctx.non_removable
        # LOCAL: None in the device assigns it a number later
        opts = BufferizeOpts(device=s.device, removable=removable) if len(ctx.range_map[s][1]) == len(realized_ranges) else \
               BufferizeOpts(device=s.device, addrspace=AddrSpace.LOCAL, removable=removable)
        new_src = UOp(Ops.STAGE, src=(new_src,)+closed_ranges, arg=opts)
        if x in ctx.range_map and not skip: new_src = new_src.index(*[r for i,r in enumerate(src_rngs) if i in realized_ranges])
    new_srcs.append(new_src)
  return new_srcs

def create_bufferize_and_index_based_on_ranges(ctx:IndexingContext, x:UOp):
  # NOTE: emitted INDEX UOps aren't in range_map, the generic path is a no-op for them
  if x.op is Ops.STAGE: return None
  shaped = x.op is Ops.INDEX and len(x.src) > 1 and any(s.shape != () for s in x.src[1:])
  new_srcs = create_bufferize_and_index_srcs(ctx, x)
  # a gather INDEX on a flat buffer re-indexes to itself: it IS the emission, don't wrap it in itself
  if any(ns is x for ns in new_srcs): return None
  ret = x.replace(src=tuple(new_srcs))
  if x in ctx.gather_views:
    # a rebuilt view-chain node no longer matches ctx.gather_views by identity: carry the marker
    # on the tag so remove_movement_op_after_rangeify keeps it for lower_shaped_gather
    ret = ret.rtag("gather_view")
  elif shaped and x in ctx.range_map:
    # hand the recorded output ranges to lower_shaped_gather. the coords may still scalarize in
    # later rebuilds, so key by tag: a rebuild keeps the tag even as its srcs are finalized
    key = f"shaped_gather{next(ctx.gather_tag)}"
    ctx.shaped_kept[key] = ctx.range_map[x][1]
    ret = ret.rtag(key)
  return ret

def convert_pad_to_where_to_keep_behavior_local(ctx:IndexingContext, x:UOp):
  # inside a gather's view chain the pad stays a PAD: lower_shaped_gather folds it into the read
  # as valid-gating (the padded cells read 0), which preserves the padded output frame
  if x not in ctx.range_map or x in ctx.gather_views: return None
  bx = create_bufferize_and_index_based_on_ranges(ctx, x)
  valid: UOp = UOp.const(True).uprod([r.get_valid() for r in ctx.range_map[x][0]])
  return valid.where(bx.src[0], UOp.const(x.dtype.const(0)))

def convert_reduce_to_reduce_with_ranges(ctx:IndexingContext, x:UOp):
  if x.arg[1] == 0: return None
  if x not in ctx.range_map: raise RuntimeError("REDUCE has no ranges in rangeify, UOp verification failed")
  bx = create_bufferize_and_index_based_on_ranges(ctx, x)
  # input ranges
  new_ranges = list(ctx.range_map[x][0][:x.arg[1]])
  return UOp(Ops.REDUCE, src=(bx.src[0],)+tuple(new_ranges), arg=(x.arg[0], 0))

def convert_stack_to_where(ctx:IndexingContext, x:UOp):
  # only data STACKs: shape tuple STACKs aren't in range_map, the empty shape tuple is void
  if x not in ctx.range_map or x.dtype == dtypes.void: return None
  # use the src list directly, a transient STACK of mid-rangeify srcs violates the spec shape rule
  srcs = create_bufferize_and_index_srcs(ctx, x)
  r0 = ctx.range_map[x][1][0]
  ret = srcs[-1]
  for k in range(len(srcs)-2, -1, -1): ret = r0.eq(k).where(srcs[k], ret)
  return ret

def remove_movement_op_after_rangeify(ctx:IndexingContext, x:UOp):
  # originals are in range_map. a movement node rebuilt after its srcs were rewritten (e.g. over
  # a STACK that became a WHERE) is structurally identical and must strip too: its mapping
  # already lives in the consumers' ranges, and a surviving reshape no longer matches its src
  if x.tag == "gather_view" or x in ctx.gather_views: return None  # the gather's view chain is folded by lower_shaped_gather
  if x in ctx.range_map or x.src[0].op is Ops.INDEX: return x.src[0]
  if x.src[0].op not in {Ops.PARAM, Ops.BUFFER, Ops.MSTACK, Ops.MSELECT, Ops.AFTER, Ops.STAGE, Ops.CAST, Ops.BITCAST}:
    return x.src[0]

# after apply, a gather (marked INDEX) whose buffer src consumed the address is a pure wrapper
# and goes away. view-ish buffer srcs keep the INDEX, later passes fold those (pm_mops)
# after apply, an INDEX whose buffer is an emitted access (an INDEX, casts commute with the
# gather) is a pure wrapper: the address already folded into that access, the node goes away
def lower_gather_index(ctx:IndexingContext, x:UOp) -> UOp|None:
  buf = x.src[0]
  while buf.op in {Ops.CAST, Ops.BITCAST}: buf = buf.src[0]
  return x.src[0] if buf.op is Ops.INDEX else None

def lower_shaped_gather(ctx:IndexingContext, x:UOp) -> UOp|None:
  """
  Emission for a gather with tensor coordinates. The coords are per-dim positions of the buffer
  view's leading block (each src, scalar or shaped, covers one leading dim; the trailing kept
  dims stay loop-carried). Fold the coords plus the kept ranges through the buffer's view chain
  with the movement algebra and emit the flat read of the buffer root, derived from the gather's
  own block/kept split. No coordinate tensor ever passes through the movement algebra as a range,
  so reshape splits and broadcast shrinks can't mangle it.
  """
  tag = x.tag
  if x.op is not Ops.INDEX or not isinstance(tag, str) or not tag.startswith("shaped_gather") or tag not in ctx.shaped_kept: return None
  buf, coords = x.src[0], x.src[1:]
  if buf.op is Ops.INDEX: return None  # gather over a gather: the rule below owns that
  chain:list[UOp] = []
  root = buf
  while root.op in GroupOp.Movement:
    chain.append(root)
    root = root.src[0]
  # kept ranges: the trailing entries of this gather's recorded output ranges. the coords may
  # have been scalarized by apply since they were spliced, so slice from the end
  nkept = len(buf.shape) - len(coords)
  if nkept < 0: return None
  out_rngs = ctx.shaped_kept[tag]
  idxs:list[UOp] = list(coords) + list(out_rngs[len(out_rngs)-nkept:])
  if len(idxs) != len(buf.shape): return None
  for m in chain:
    idxs = list(apply_movement_op(m.op, m.src[0].shape, m.marg, tuple(idxs)))
  return root.index(*idxs)
pm_lower_gathers = PatternMatcher([
  (UPat(Ops.INDEX, name="x", allow_any_len=True), lower_shaped_gather),
  (UPat(Ops.INDEX, name="x", allow_any_len=True), lower_gather_index)])


pm_apply_rangeify = PatternMatcher([
  # REDUCE(op, axis) -> REDUCE(op) with ranges
  (UPat(Ops.REDUCE, name="x"), convert_reduce_to_reduce_with_ranges),
  # PAD -> WHERE
  (UPat(Ops.PAD, name="x"), convert_pad_to_where_to_keep_behavior_local),
  # STACK -> WHERE select on the leading range
  (UPat(Ops.STACK, name="x"), convert_stack_to_where),
  # finally, apply_rangeify
  (UPat(GroupOp.All, name="x"), create_bufferize_and_index_based_on_ranges),
  # remove movement op
  (UPat(GroupOp.Movement, name="x"), remove_movement_op_after_rangeify),
])

pm_fix_deviceless = PatternMatcher([
  (UPat(Ops.STAGE, name="b"),
    lambda ctx,b: b.replace(arg=replace(b.arg, device=ctx)) if b.arg.addrspace is AddrSpace.GLOBAL and b.arg.device is None else None),
])

@functools.cache
def _apply_reshape(in_shape:tuple[sint,...], out_shape:tuple[sint, ...], urngs:UOp) -> UOp:
  acc:sint = 1
  axes_in:list[UOp] = []
  for s,src in list(zip(out_shape, urngs.src))[::-1]:
    axes_in.append(acc*src)
    acc *= s
  combined_axes = UOp.const(0).usum(axes_in)
  axes_out:list[UOp] = []
  for s in in_shape[::-1]:
    axes_out.append(combined_axes % s)
    combined_axes //= s
  # this simplify is doing a lot of heavy lifting. this is the replacement for the reshape view merging code
  return graph_rewrite(UOp.sink(*axes_out[::-1]), symbolic+pm_simplify_valid+pm_drop_and_clauses, name="reshape")

# this is the definition of the movement ops
@functools.cache
def apply_movement_op(op:Ops, in_shape:tuple[sint,...], arg:tuple, rngs:tuple[UOp, ...]) -> tuple[UOp, ...]:
  match op:
    case Ops.SHRINK:  rngs = tuple(a if off == 0 else a+off for a,(off,_) in zip(rngs, arg))
    case Ops.PERMUTE: rngs = tuple(rngs[p] for p in argsort(arg))
    case Ops.FLIP:    rngs = tuple(((s-1)-a) if f else a for a,s,f in zip(rngs, in_shape, arg))
    case Ops.EXPAND:  rngs = rngs[len(arg):]
    case Ops.PAD:
      # NOTE: the .where(r-s, i) is not inside the graph_rewrite so that `convert_pad_to_where_to_keep_behavior_local`
      #       wraps the pad with only the newly added valid
      rngs = tuple(r if (sz == sh and off == 0) else (r-off).valid(graph_rewrite((r >= off) & (r < (sh+off)),
        symbolic+pm_simplify_valid, name="pad")) for r,sh,(off,sz) in zip(rngs, in_shape, arg))
    case Ops.RESHAPE:
      sink = UOp.sink(*rngs).simplify() # NOTE: this applies any commutative flips to the rngs early
      sub_array = {r:r.replace(src=r.src[:1], arg=(i, AxisType.PLACEHOLDER)) for i,r in enumerate(sink.ranges)}
      rngs = _apply_reshape(in_shape, arg, sink.substitute(sub_array)).substitute({v:k for k,v in sub_array.items()}).src
    case _: raise RuntimeError(f"{op} is not a MovementOp")
  return rngs

@rewrite_group(new_ctx=False)
def run_rangeify(tsink:UOp, debug:bool=False) -> UOp:
  if debug: print("**************************")
  rctx = IndexingContext()

  # get ops to realize
  graph_rewrite(tsink, pm_generate_realize_map, ctx=rctx, name="get realize")

  # get the consumer map
  with cpu_profile("consumer map in rangeify", "TINY"):
    tsink_toposort = tsink.toposort(gate_kernel_sink)
    consumer_map: dict[UOp, dict[UOp, None]] = {x:{} for x in tsink_toposort}
    for c in tsink_toposort:
      for x in data_srcs(c.op, c.src):
        if x in consumer_map: consumer_map[x][c] = None

  # explicit rangeify
  ending_ranges: dict[UOp, list[UOp]] = {}
  for x in reversed(tsink_toposort):
    # no ranges on kernels, they are internal
    if x.op in {Ops.CALL, Ops.FUNCTION, Ops.LINEAR}: continue

    # AFTER doesn't have range
    if x.op is Ops.AFTER: continue

    # treat MSTACK/MSELECT like SINK
    if x.op in {Ops.MSTACK, Ops.MSELECT}: continue

    # NOTE: ending ranges don't propagate into the address of a gather (an INDEX index src), the
    # address is consumed elementwise and shouldn't force a realize of its computation
    ending_ranges[x] = sum([ending_ranges.get(u, []) for u in consumer_map[x]
                            if not (u.op is Ops.INDEX and x in u.src[1:])], [])
    # ranges the consumers iterate that this node broadcasts over
    ended = [rctx.range_map[c][0][i] for c in consumer_map[x] if c in rctx.range_map and c.op in GroupOp.Broadcastable
             for i in broadcast_axes(x.shape, c.shape)]
    broadcast_ending_ranges = list(UOp.sink(*ended).ranges)
    # fusion decision: REDUCE before the broadcast
    if x.op is Ops.REDUCE: ending_ranges[x] += broadcast_ending_ranges

    # *** the ranges on the output are
    #  1. new if this op is realized
    #  2. from the single consumer if this op only has one consumer
    #  3. potentially new if this op has 2+ consumers

    consumer_rngs = []
    for c in consumer_map[x]:
      if c not in rctx.range_map: continue
      # an index src of a shaped INDEX broadcasts against the gathered output dims like an ALU operand
      if c.op is Ops.INDEX and len(c.src) > 1 and x in c.src[1:] and x.shape != ():
        big_shape = _broadcast_shape(*(s.shape for s in c.src[1:]))
        baxes, nleft = broadcast_axes(x.shape, big_shape), len(big_shape)-len(x.shape)
        out_rngs_c = rctx.range_map[c][1]
        consumer_rngs.append(tuple(r.const_like(0) if j in baxes else r for j,r in enumerate(out_rngs_c[:len(big_shape)]) if j >= nleft))
      # a gather view's frame ranges are placeholder bookkeeping for the emission fold
      # (lower_shaped_gather), not real loops over x: they must not drive range merges, two gathers
      # over one view would otherwise realize the view into a STAGE the flat read never targets
      elif c.op is Ops.INDEX and len(c.src) > 1 and c.src[0] is x and x in rctx.gather_views:
        continue
      else: consumer_rngs.append(broadcast_rngs(c, x, rctx.range_map[c][0]))
    if x in rctx.realize_map:
      # if this is in the realize_map, we create new ranges (at the output)
      out_rngs = tuple(rctx.new_range(s) for s in x.shape)
      # all ranges are ended now
      ending_ranges[x] = []
      # mark all ranges as ended
      assert rctx.realize_map[x] is None
      rctx.realize_map[x] = list(range(len(x.shape)))
    elif len(consumer_rngs) == 0:
      # if no consumers have ranges and this isn't realized, this doesn't have ranges either.
      continue
    elif len(consumer_rngs) == 1:
      # if this has one consumer, it inherits the ranges from it
      out_rngs = consumer_rngs[0]
    elif len(consumer_rngs) > 1:
      # if this has two consumers, we have to merge the ranges and might create new ones
      all_rngs: list[tuple[UOp, ...]] = list(zip(*consumer_rngs))
      rngs_valids = []
      for valid_rngs in all_rngs:
        local_rngs, valids = zip(*[(r.get_idx(), r.get_valid()) for r in valid_rngs])
        rngs_valids.append((local_rngs, valids))

      # TODO: in RANGEIFY > 1 all_all_same isn't required
      all_all_same = all(all_same(local_rngs) for local_rngs,_ in rngs_valids)
      _out_rngs = []
      _realize_axis = []
      for i,(local_rngs,valids) in enumerate(rngs_valids):
        # we compare the ranges without their valids
        if all_all_same or (PCONTIG and all_same(local_rngs)):
          # the new valid is the OR of all the children valids
          minimum_valid = UOp.const(False).usum(valids)
          _out_rngs.append(graph_rewrite(local_rngs[0].valid(minimum_valid), symbolic, name="minimum_valid"))
        else:
          _out_rngs.append(rctx.new_range(x.shape[i]))
          _realize_axis.append(i)
      out_rngs = tuple(_out_rngs)

      # we have to (partially) realize here if there's new ranges
      if len(_realize_axis): rctx.realize_map[x] = _realize_axis

    # if this element is a reduce and there's ended ranges, we might have to end some other ranges
    if len(ending_ranges[x]) and x.op in GroupOp.Elementwise.union({Ops.REDUCE}):
      _realize_axis = rctx.realize_map.get(x) or []
      for i,r in enumerate(out_rngs):
        if i in _realize_axis: continue
        if not (PCONTIG > 1) or any(any(rr.arg > e.arg for e in ending_ranges[x]) for rr in r.ranges):
          _realize_axis.append(i)
      ending_ranges[x] = []
      if len(_realize_axis):
        rctx.realize_map[x] = _realize_axis
        out_rngs = tuple([(rctx.new_range(x.shape[i]) if i in _realize_axis else r) for i,r in enumerate(out_rngs)])
    ending_ranges[x] += broadcast_ending_ranges

    # TODO: some ops don't have shape, enable this after the `.st` property is removed
    #assert len(out_rngs) == len(x.shape), \
    #  f"shape len mismatch {len(out_rngs)} != {len(x.shape)} on {x.op} with {len(consumer_map[x])} consumers and realize {x in realize_map}"

    # *** the ranges on the inputs are
    #  1. swizzled for MovementOps
    #  2. newly created for REDUCE (tensor graph form with axis)
    #  3. passed through for everything else

    rngs = out_rngs  # rngs is the input ranges  # pylint: disable=possibly-used-before-assignment

    # a gather INDEX covers the leading dims of its buffer with its coordinate srcs instead of
    # ranges (a scalar coordinate still selects one leading axis). a producer containing a REDUCE
    # can't fold the address (the reduce machinery works on ranges): realize its data root and
    # read the gather flat. the toposort gates at AFTER, the provenance of a realized buffer is
    # already materialized
    if x.op is Ops.INDEX and len(x.src) > 1:
      buf = x.src[0]
      while buf.op in GroupOp.Movement: buf = buf.src[0]
      if buf not in rctx.realize_map and any(n.op is Ops.REDUCE for n in buf.toposort(gate=lambda n: n.op is not Ops.AFTER)):
        rctx.realize_map[buf] = None
        # the address can't fold through this producer, the STAGE has to survive removal
        rctx.non_removable[buf] = None
      if any(s.shape != () for s in x.src[1:]) and buf.op not in ALWAYS_CONTIGUOUS | {Ops.INDEX} \
        and buf not in rctx.realize_map:
        # a tensor-coordinate gather reads its buffer root flat, so a computed (non-buffer) root
        # must materialize: the flat read targets the STAGE, not the value's producer
        rctx.realize_map[buf] = None
        rctx.non_removable[buf] = None
      for s in x.src[1:]:
        # a coordinate produced through state (an AFTER) can't fuse its production with the
        # gather's ranges: realize it so the stateful producer stays its own kernel
        root = s
        while root.op in GroupOp.Movement: root = root.src[0]
        if root not in rctx.realize_map and any(n.op is Ops.AFTER for n in root.toposort(gate=lambda n: n.op is not Ops.AFTER)):
          rctx.realize_map[root] = None
          rctx.non_removable[root] = None
      if any(s.shape != () for s in x.src[1:]):
        # a tensor coordinate is data flow, not a range: spliced into the rng tuple it would be
        # broadcast/reshaped by the movement algebra like an ordinary operand (zeroed on shrunk
        # axes, mis-weighted by reshape splits). give the covered leading dims placeholder
        # ranges so the view chain's bookkeeping stays shape-consistent, keep the view chain out
        # of apply-time indexing, and lower the gather to a flat load at emission instead
        # (lower_shaped_gather)
        rngs = tuple(rctx.new_range(s) for s in x.src[0].shape[:len(x.src)-1]) \
             + out_rngs[len(_broadcast_shape(*(s.shape for s in x.src[1:]))):]
        view = x.src[0]
        while view.op in GroupOp.Movement:
          rctx.gather_views.add(view)
          view = view.src[0]
      else:
        # scalar coords cover one leading dim each and are honest rng entries
        rngs = tuple(x.src[1:]) + out_rngs[len(_broadcast_shape(*(s.shape for s in x.src[1:]))):]

    # apply movement ops
    if x.op in GroupOp.Movement: rngs = apply_movement_op(x.op, x.src[0].shape, x.marg, rngs)
    # STACK: the leading range selects the src, srcs get the trailing ranges
    if x.op is Ops.STACK: rngs = out_rngs[1:]
    # if the EXPAND is used to inject a range, we don't mark it as ending_ranges. otherwise we do.
    # NOTE: this doesn't actually always end a range, but this is why convs are realized, so for now we need it
    if x.op is Ops.EXPAND and all(isinstance(y, int) or y.op is not Ops.RANGE for y in x.shape):
      ending_ranges[x] += list(UOp.sink(*out_rngs[:len(x.marg)]).ranges.keys())

    # REDUCE creates ranges for the axes it is reducing
    if x.op is Ops.REDUCE and x.arg[1]:
      rngs = tuple(rctx.new_range(s, axistype=AxisType.REDUCE) for s in x.src[0].shape[:x.arg[1]]) + out_rngs

    if debug:
      realized_ranges = rctx.realize_map.get(x, None)
      if x.op is Ops.RESHAPE or len(rngs) != len(out_rngs):
        disp = render_ranges(rngs, realized=realized_ranges) + " -> " + render_ranges(out_rngs, realized=realized_ranges)
      else:
        disp = render_ranges(rngs, out_rngs, realized=realized_ranges)
      print("***" if x in rctx.realize_map else "   ",
            f"{len(consumer_map[x]):2d} {str(x.op):20s} {str(x._shape):35s} {len(ending_ranges[x]):2d}", disp)

    # assign to the range map. rngs are the input ranges, out_rngs are the output ranges, from the x op.
    rctx.range_map[x] = (rngs, out_rngs)

  # NOTE: SPEC=3 is broken here with shape
  with Context(SPEC=min(SPEC.value, 2)):
    tsink = graph_rewrite(tsink, pm_apply_rangeify, ctx=rctx, bottom_up=True, name="apply rangeify")
    tsink = graph_rewrite(tsink, pm_lower_gathers, ctx=rctx, name="lower gathers")
  # if a deviceless value must materialize, place it on the sink device
  tsink = graph_rewrite(tsink, pm_fix_deviceless, ctx=tsink.device, name="add device to deviceless")
  return tsink

def render_ranges(*rngs_list, realized) -> str:
  disp = []
  for i, rs in enumerate(zip(*[[r.render() for r in rngs] for rngs in rngs_list])):
    rng = rs[0] if all_same(rs) else " -> ".join(rs)
    if realized is not None and i in realized: rng = colored(rng, "yellow")
    disp.append("["+rng+"]")
  return ''.join(disp)
