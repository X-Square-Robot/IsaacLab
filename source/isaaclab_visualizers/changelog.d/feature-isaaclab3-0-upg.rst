Fixed
^^^^^

* Fixed ``NameError: name 'advancing' is not defined`` raised inside
  :meth:`~isaaclab_visualizers.newton.newton_visualizer.NewtonVisualizer.step`.
  A merge had dropped the ``advancing = dt > 0.0`` definition while keeping its
  two uses (the update-frequency throttle and the pending single-step consume),
  so the Newton visualizer crashed on every step. The definition is restored.
