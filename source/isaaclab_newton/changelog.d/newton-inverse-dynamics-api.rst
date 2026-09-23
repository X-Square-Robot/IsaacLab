Changed
^^^^^^^

* Changed the StablePD controller-state assembly to Newton's new
  :class:`newton.InverseDynamics` API: a single ``eval_inverse_dynamics``
  call with ``EvalType.ALL`` now provides the mass matrix and the
  gravity/Coriolis bias, replacing the removed
  ``eval_inverse_dynamics(joint_acc=None)`` overload and the manual
  ``eval_jacobian`` / ``eval_mass_matrix`` scratch buffers.
