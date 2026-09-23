Fixed
^^^^^

* Fixed :meth:`~isaaclab.sim.schemas.modify_articulation_root_properties` crashing in
  kitless runs (Newton / OvPhysX physics without Omniverse Kit) when
  :attr:`~isaaclab.sim.schemas.ArticulationRootPropertiesCfg.fix_root_link` is ``True`` and
  the USD has no pre-authored fixed joint. The fixed-joint creation path imported the
  Kit-only ``omni.physx.scripts.utils``, raising ``ModuleNotFoundError: No module named
  'omni.physx'``. It now authors the world-to-root :class:`pxr.UsdPhysics.FixedJoint`
  (with full local poses) directly via USD physics when Kit is unavailable, so fixed-base
  articulations can be spawned on kitless backends; the Kit path is unchanged.
