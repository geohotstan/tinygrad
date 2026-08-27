# Tensor indexing: state of the direct rewrite

Branch `direct-indexing` carries an in-progress rewrite of tensor indexing onto a
"loop and pick" engine. This note records the design, what works, and the exact
remaining work, so the next session starts at a boundary instead of an open question.

## The target design

Indexing is looping: run over the positions of the output, pick the source
coordinates listed in the index expressions, and load (or store) exactly the
element at those coordinates. No one-hot masks over class counts, no combining
index tensors into one flat linear address at trace time, no scatter masking.

## What is landed (all kernel-proven)

- `_commit_writes` (mixin/op.py): the shared write commit for advanced setitem,
  scatter and scatter_reduce. Per-position coordinate streams (tensor group shares
  one pairwise slot-block, slice windows cartesian), duplicate destinations summed
  or last-wins among the writes themselves, out-of-bounds drops via a sacrificial
  staging cell, single ordered indexed store through a custom_kernel.
- `index_gradient` (mixin/gradient.py): gather backward as an eager scatter-sum
  kernel; duplicates sum, OOB drops. Registered for Ops.INDEX in pm_gradient.
- Scheduler groundwork (schedule/indexing.py): scalar-coordinate shaped gathers,
  WAR serialization for same-anchor assign writers.

## The open blocker: shaped-INDEX emission

The read side still needs an emission contract. Today the frontend emits
`INDEX(flat_view, addr)` where `flat_view.shape = (block,) + kept` and
`addr.shape = big` (concat rule: output = big ++ kept). Scheduling this requires
coordinate tensors inside range tuples, which movement algebra cannot consume:
it builds malformed RESHAPE/PAD nodes that crash codegen
(`expand_broadcast`/`do_devectorize` shape mismatches like () -> (3,1) or
((2,3,4,5,6,1),(4,3,2))).

The fix: a replace-semantics emitter at the tail of run_rangeify that lowers each
shaped INDEX into per-dim address decomposition at emission time (no rng splicing,
no slot bookkeeping). Concretely: an `Ops.INDEX` whose coordinate srcs are shaped
is rewritten into a flat load against the same flattened view, with the flat
address expressed as movement algebra over the coordinate values and the kept
ranges -- derived from the gather's own recorded block/kept split, never from
view-chain introspection (expanded or None-injected views must not participate).

First validation gate: `test_slice_fancy_indexing_dim_collapse_int`.
Then: delete the one-hot `gather` and the flatten-combined `_getitem` frontend,
swap both onto the pick core, run
`SPEC=2 DEV=NULL pytest -n12 test/null/`,
`pytest -n12 test/unit/test_indexing.py test/backend/test_setitem.py
"test/backend/test_ops.py::TestOps" -k 'gather or scatter or fancy_indexing'`,
`python -m mypy tinygrad/`, `python -m ruff check`.

## Known failing set at bb17ff8af

Eight fancy-shape tests (crash in `expand_broadcast`/`do_devectorize`),
`test_index_ind_dtype` (coalesce multi-store), the null tensor_uop_mixin bucket,
and ~14 mypy arg-type errors in mixin/op.py. Everything else is green.
