Fixed
^^^^^

* Fixed ``Simulation view object is invalidated`` errors and stale articulation views
  on the PhysX backend when :mod:`isaaclab_physx` is imported before Kit starts
  (e.g. for pre-launch config loading). In that import order Isaac Sim's own
  ``SimulationManager`` default warm-start callback stayed enabled and force-reloaded
  physics from USD during reset, deleting the warmed-up PhysX scene under the manager's
  simulation views. :meth:`~isaaclab_physx.physics.PhysxManager.initialize` now disables
  those default callbacks through the official ``enable_all_default_callbacks`` API once
  Kit is running.
