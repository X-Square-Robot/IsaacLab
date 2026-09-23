Added
^^^^^

* Added :attr:`~isaaclab.actuators.StablePDActuatorCfg.gravity_compensation` selecting the
  StablePD gravity-compensation scheme per actuator group: ``"feedforward"`` (default) adds the
  full-weight gravity torque through the controller's constant-effort channel (zero steady-state
  droop, effort-clamped), ``"bias"`` feeds ``g(q)`` into the implicit predictor's bias (partial
  compensation, the native controller convention), and ``"none"`` disables it.
