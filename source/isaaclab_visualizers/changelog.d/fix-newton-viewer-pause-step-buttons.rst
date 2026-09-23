Fixed
^^^^^

* Fixed the Newton viewer's top-panel ``Pause`` / ``Step`` buttons and the ``Space`` / ``.``
  keyboard shortcuts being inert. Newton drives them through its ``should_step`` protocol, which
  the Isaac Lab step loop never calls, so they had no effect. They are now bridged to Isaac Lab's
  pause gate: ``Pause`` and ``Space`` toggle the "Pause Simulation" state, and ``Step`` / ``.``
  advance exactly one physics frame while paused.
