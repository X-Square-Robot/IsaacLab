Fixed
^^^^^

* Fixed :class:`~isaaclab.app.AppLauncher` leaving duplicate Isaac Lab modules in ``sys.modules``
  after Kit startup. Modules freshly imported by Kit extensions while the launcher temporarily
  hid the pre-loaded Isaac Lab modules are now dropped before the originals are restored, so
  configuration classes no longer exist as two distinct copies. Previously this broke
  ``isinstance`` checks across the twins, e.g. spawning a ground plane with the default physics
  material on the PhysX preset raised ``TypeError`` in
  :func:`~isaaclab.sim.spawners.materials.spawn_physics_material`.
