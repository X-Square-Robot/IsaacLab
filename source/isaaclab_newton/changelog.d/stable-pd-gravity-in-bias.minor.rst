Changed
^^^^^^^

* Changed the Newton fast-path StablePD actuator to feed gravity into the
  ``ControllerStablePD`` bias forces (Tan 2011 ``C = -tau_g + tau_c``) instead of
  applying it as a full-weight feedforward on ``control.joint_f`` after the actuator
  step, matching Newton's StablePD reference path. Gravity now
  flows through the controller's implicit solve and the ``ClampingMaxEffort`` limit
  (it no longer bypasses the effort clamp); as in the example the arm partially
  droops under gravity. The Coriolis finite difference also no longer re-runs
  ``eval_fk`` after each perturbation (``eval_jacobian`` reads the perturbed
  ``joint_q`` directly), removing ``num_joints`` forward-kinematics passes per
  substep.
