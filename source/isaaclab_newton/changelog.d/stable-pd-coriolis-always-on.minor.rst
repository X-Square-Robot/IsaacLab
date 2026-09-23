Changed
^^^^^^^

* Changed the Newton fast-path StablePD actuator to always assemble the
  Coriolis/centrifugal term ``tau_c`` into the Tan 2011 bias forces
  (finite-difference of ``M(q)`` with the Christoffel contraction), matching
  Newton's StablePD reference path. Previously ``tau_c`` was
  only computed when the ``MANAENV_STABLEPD_CORIOLIS`` environment variable was
  set; that gate is removed and the variable is no longer read. Gravity is
  unchanged (still applied full-weight in ``joint_f``); ``tau_c`` is ~0 at rest
  and only contributes under fast motion.
