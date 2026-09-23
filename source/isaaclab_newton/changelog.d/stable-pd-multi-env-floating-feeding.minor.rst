Added
^^^^^

* Added multi-env and floating-base support to StablePD mass-matrix feeding:
  per-env gather kernels and a 6-DOF base Schur reduction behind the new
  CUDA-graph-capturable ``StablePDFeeder``, wired into the Newton articulation
  feeding path (``num_base_dofs`` derived from the articulation base type).

Fixed
^^^^^

* Fixed ``ControllerStablePD`` being constructed with ``num_worlds=1`` when
  actuators are built from USD outside a Newton model (the PhysX adapter
  path), which folded all envs into a single implicit solve. The adapter now
  wires the replicated env count into controllers that accept ``num_worlds``.
