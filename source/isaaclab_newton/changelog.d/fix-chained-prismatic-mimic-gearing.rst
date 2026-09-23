Fixed
^^^^^

* Fixed :meth:`~isaaclab_newton.physics.NewtonManager._normalize_physx_angular_mimic_gearing`
  skipping the second finger of a chained parallel-gripper mimic, leaving one
  jaw under-driven by ``180/pi`` (~57.3x). The Newton USD importer keys a
  follower's angular-ness off the ``physxMimicJoint`` axis-instance name
  (``rot*`` vs ``trans*``); an EX001-style gripper authors its prismatic finger
  joints on a ``rot*`` axis, so a finger-to-finger coupling meant to be a
  dimensionless ``-1.0`` is stored as ``-pi/180``. The gearing normalization now
  also rescales a prismatic follower geared off a prismatic leader (not only a
  revolute leader), so both jaws track their leader symmetrically under MuJoCo
  Warp.
