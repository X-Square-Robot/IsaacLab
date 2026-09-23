Changed
^^^^^^^

* Changed the Newton fast-path StablePD actuator hook to be CUDA-graph capturable:
  its per-substep ``ControllerStablePD`` input assembly (mass matrix, Jacobian-transpose
  gravity, finite-difference Coriolis) now runs entirely as ``wp.launch`` / ``wp.copy`` /
  Newton ``eval_*`` into pre-allocated scratch, replacing the Python-guarded lazy
  ``Articulation.data`` accessors that forced eager execution. When every actuator is
  graph-safe the full ``decimation x (actuators + solver)`` loop is now captured into a
  single CUDA graph on the StablePD path too. :meth:`NewtonManager.register_pre_actuator_callback`
  gained a ``graphable`` flag so a hook can opt into capture.

Fixed
^^^^^

* Fixed the Newton fast-path StablePD Coriolis term being identically zero. The
  finite-difference of ``M(q)`` reads body frames via ``eval_jacobian`` / ``eval_mass_matrix``,
  which only ``eval_fk`` updates, so without an ``eval_fk`` after each joint perturbation
  ``dM/dq`` (and therefore ``tau_c``) collapsed to zero. The forward-kinematics pass is
  now run per perturbation so the Coriolis/centrifugal bias is actually computed.
