Fixed
^^^^^

* Added per-articulation Newton actuator selection through
  :attr:`~isaaclab.assets.ArticulationCfg.native_actuators`. Existing
  configurations inherit :attr:`~isaaclab.sim.SimulationCfg.use_newton_actuators`;
  set the articulation field to ``False`` to keep an asset on the standard
  Isaac Lab actuator path.
