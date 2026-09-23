Fixed
^^^^^

* Fixed :func:`~isaaclab_newton.actuators.build_implicit_dof_mask` to accept actuator
  ``joint_indices`` given as plain integer lists (as produced by the OvPhysX backend)
  in addition to torch tensors.
