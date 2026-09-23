Changed
^^^^^^^

* Changed :class:`~isaaclab_ovphysx.physics.OvPhysxManager` to load scenes through
  the ``ovstage`` attach API introduced by public ``ovphysx`` 0.5.9, while retaining
  compatibility with the legacy 0.4.13 USD-handle API. Existing scene setup calls
  require no changes; reinstall the ``ov[ovphysx]`` extra when upgrading an
  environment in place.

* Changed OvPhysX clone replay to forward stable environment ids when supported by
  the 0.5.9 runtime, preserving collision filtering across heterogeneous clone calls.

Fixed
^^^^^

* Fixed remote USD asset loading with ``ovphysx`` 0.5.9 by initializing its
  matched OVStage/OmniClient runtime before another ``omniverseclient`` wheel can
  load an incompatible process-global library.
