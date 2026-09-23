Fixed
^^^^^

* Fixed the Newton-actuator telemetry launch in
  :class:`~isaaclab_ovphysx.assets.Articulation` to pass the joint-ordering
  arguments added to the shared ``sync_torque_telemetry`` kernel, which made
  every ``write_data_to_sim`` call fail with a kernel-argument-count error.
