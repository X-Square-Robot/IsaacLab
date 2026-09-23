Changed
^^^^^^^

* Changed the ``ovphysx`` pin in the root ``ov`` extra and
  ``[tool.isaaclab.versions]`` from the internal ``0.5.2+head.f62c22207c`` build to
  the public ``0.4.13`` wheel available on ``pypi.nvidia.com``, so
  ``./isaaclab.sh -i 'ov[ovphysx]'`` resolves without access to internal package
  registries. The OvPhysX manager gained the corresponding API compatibility
  fallbacks.
