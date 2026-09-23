Fixed
^^^^^

* Fixed ``NameError: name 'LiftSimCfg' is not defined`` raised at import time of
  :mod:`isaaclab_tasks.core.lift.lift_env_cfg`. A merge had dropped the
  ``LiftSimCfg`` preset class definition while keeping the ``sim: LiftSimCfg =
  LiftSimCfg()`` reference on :class:`~isaaclab_tasks.core.lift.lift_env_cfg.LiftEnvCfg`,
  breaking every entry point that imports the lift task. The PhysX and Newton
  simulation presets were restored.
