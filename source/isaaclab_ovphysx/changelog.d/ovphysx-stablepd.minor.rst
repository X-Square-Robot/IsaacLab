Added
^^^^^

* Added the Newton actuator fast path (``use_newton_actuators=True``) to the OvPhysX
  :class:`~isaaclab_ovphysx.assets.Articulation`, enabling
  :class:`~isaaclab.actuators.StablePDActuatorCfg` on the OvPhysX backend. StablePD
  controllers are fed the wheel's joint-space mass matrix, Coriolis, and gravity
  bindings each step, mirroring the PhysX backend's controller path. Fixed-base
  articulations only; floating-base support raises ``NotImplementedError``.
