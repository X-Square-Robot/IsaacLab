Fixed
^^^^^

* Fixed :meth:`~isaaclab_newton.physics.NewtonManager.instantiate_builder_from_stage`
  leaving ``_num_envs`` unset on the flat-loading path (no ``/World/Env_*``
  Xforms), which raised ``TypeError: unsupported operand type(s) for //: 'int'
  and 'NoneType'`` in
  :meth:`~isaaclab_newton.physics.NewtonManager.activate_newton_actuator_path`
  when the Newton actuator fast path (``use_newton_actuators=True``) ran on a
  single-world stage. The flat path now sets ``_num_envs`` to ``1``.
