Added
^^^^^

* Added per-actuator gravity-compensation modes to the StablePD feeding path. The feeder now
  honors :attr:`~isaaclab.actuators.StablePDActuatorCfg.gravity_compensation`: ``"feedforward"``
  (default) gathers ``g(q)`` into the controller's ``const_effort`` channel (full weight,
  effort-clamped, CUDA-graph capturable), ``"bias"`` keeps it in the implicit bias, and
  ``"none"`` gathers the Coriolis term only. On a floating base the 6 base rows keep their
  gravity term in every mode.
