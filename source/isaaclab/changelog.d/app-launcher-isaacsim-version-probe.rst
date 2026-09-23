Fixed
^^^^^

* Fixed :meth:`~isaaclab.app.AppLauncher.is_isaac_sim_version_5` raising an opaque
  ``NameError: name 'isaacsim' is not defined`` when Isaac Sim is not importable. The
  module-level import is suppressed, so the probe now re-imports locally and raises an
  ``ImportError`` naming the missing package and the usual causes.
