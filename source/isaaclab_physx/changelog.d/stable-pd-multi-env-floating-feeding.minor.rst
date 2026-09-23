Added
^^^^^

* Added multi-env and floating-base support to the PhysX StablePD
  mass-matrix feeding: the per-actuator joint block is sliced past the 6
  prepended root DOFs and the unactuated base block is Schur-eliminated
  before feeding ``ControllerStablePD``, matching the Newton backend's
  ``StablePDFeeder`` semantics. The previous single-env PoC warnings are
  removed.
