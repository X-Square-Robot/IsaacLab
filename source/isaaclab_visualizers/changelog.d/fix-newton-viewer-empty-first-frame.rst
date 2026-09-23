Fixed
^^^^^

* Fixed :class:`~isaaclab_visualizers.newton.NewtonVisualizer` rendering an empty
  viewport on startup. The viewer starts paused, and the paused branch of
  :meth:`~isaaclab_visualizers.newton.NewtonVisualizer.step` only redraws existing
  instance buffers without logging state, so the initial frame held no body
  transforms until the user resumed. The first frame is now always drawn through
  the full ``log_state`` path so the scene is visible immediately.
