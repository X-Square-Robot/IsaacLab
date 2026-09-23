Fixed
^^^^^

* Fixed :class:`~isaaclab_physx.sim.spawners.materials.PhysxRigidBodyMaterialCfg` missing its
  ``_usd_namespace``, which made spawning any asset that carries a PhysX rigid-body material
  raise ``ValueError: PhysxRigidBodyMaterialCfg declares fields [...] but does not define
  '_usd_namespace'``. :func:`~isaaclab.sim.schemas.apply_namespaced` groups writes by the class
  declaring each field and reads the metadata from that class's own ``__dict__``, so the
  ``physics`` namespace inherited from
  :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg` did not cover the
  ``PhysxMaterialAPI`` fields declared on the subclass. The class now declares
  ``physxMaterial``, matching
  :class:`~isaaclab_physx.sim.spawners.materials.PhysxMaterialCfg`.
