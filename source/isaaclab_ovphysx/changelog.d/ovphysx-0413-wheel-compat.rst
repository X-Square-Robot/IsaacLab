Fixed
^^^^^

* Fixed :class:`~isaaclab_ovphysx.physics.OvPhysxManager` to also run against the
  public ``ovphysx`` 0.4.13 wheel: device selection falls back to the ``device``
  constructor argument when ``PhysX.set_cpu_mode`` is unavailable, stage clearing
  falls back from ``reset_stage()`` to ``reset()``, and ``step_sync`` passes the
  legacy ``sim_time`` argument when the wheel requires it.
