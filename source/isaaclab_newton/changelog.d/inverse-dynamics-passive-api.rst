Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.assets.articulation.StablePDFeeder` crashing on
  ``Model.inverse_dynamics`` after Newton removed the inverse-dynamics container
  (newton 95a1cb9b): the feeder now pre-allocates its own mass-matrix and
  gravity/Coriolis output buffers and calls ``newton.eval_inverse_dynamics_passive``.
