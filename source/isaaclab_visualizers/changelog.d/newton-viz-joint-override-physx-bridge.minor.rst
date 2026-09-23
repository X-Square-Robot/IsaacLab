Added
^^^^^

* Added :meth:`~isaaclab_visualizers.newton.NewtonVisualizer.get_joint_override_targets` exposing
  the Joint Control panel's slider targets as ``(USD joint prim path, value)`` pairs. On the PhysX
  sim backend the panel previously wrote into the shadow Newton control that the simulation never
  reads, leaving the robot undriven while the override muted the app's own commands; app loops can
  now bridge the panel targets into their articulation (``set_joint_position_target`` +
  ``write_data_to_sim``) each step.
