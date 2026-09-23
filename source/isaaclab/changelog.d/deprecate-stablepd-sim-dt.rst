Deprecated
^^^^^^^^^^

* Deprecated :attr:`~isaaclab.actuators.StablePDActuatorCfg.sim_dt`; it is no
  longer used and has no effect. The Tan 2011 predictor is evaluated by Newton's
  in-graph :class:`~newton.actuators.ControllerStablePD`, which reads the physics
  sub-step dt directly. Remove the argument; it can be dropped safely.
