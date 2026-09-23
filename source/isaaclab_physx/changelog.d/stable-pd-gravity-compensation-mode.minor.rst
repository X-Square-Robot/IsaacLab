Added
^^^^^

* Added per-actuator gravity-compensation modes to the PhysX StablePD feeding path, honoring
  :attr:`~isaaclab.actuators.StablePDActuatorCfg.gravity_compensation`: ``"feedforward"``
  (default) gathers ``g(q)`` into the controller's ``const_effort`` channel (full weight,
  effort-clamped), ``"bias"`` keeps it in the implicit bias, and ``"none"`` feeds the Coriolis
  term only. A robot spawned with ``disable_gravity=True`` still forces ``"none"`` everywhere
  (PhysX's gravity query ignores the per-body flag).
