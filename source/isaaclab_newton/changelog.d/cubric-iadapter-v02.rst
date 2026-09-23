Fixed
^^^^^

* Fixed the cubric ctypes shim rejecting the ``omni::cubric::IAdapter`` version shipped with Kit
  110.1.2, which made every Kit-viewport run log ``cubric IAdapter version incompatible`` and fall
  back to the CPU ``update_world_xforms()`` hierarchy pass. ``IAdapter`` moved from v0.1 to v0.2
  (appending one method at the end of the vtable and turning ``bindToStage`` into a compatibility
  thunk that supplies the new third argument), so the existing offsets and call signatures remain
  valid; the shim now accepts both versions instead of pinning a single one. Restoring the GPU path
  matters at scale: the propagation call is flat in prim count (~0.57 ms) where the CPU pass is
  linear (~3.3 us/prim), so the two cross over at roughly 120 prims.
