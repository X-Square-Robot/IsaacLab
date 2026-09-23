Fixed
^^^^^

* Fixed noisy startup logging for fixed-base assets: the missing root-com-velocity
  message in :class:`~isaaclab_newton.assets.ArticulationData` and
  :class:`~isaaclab_newton.assets.RigidObjectData` is expected for fixed-base
  assets and now logs at debug level, keeping the warning only for the
  unexpected floating-base case.
