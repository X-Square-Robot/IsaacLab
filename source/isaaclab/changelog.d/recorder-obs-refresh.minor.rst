Added
^^^^^

* Added :attr:`~isaaclab.envs.ManagerBasedRLEnvCfg.recorder_obs_refresh` to skip the extra
  observation computation before recorder post-step terms when no active term reads
  ``obs_buf`` post-step. Defaults to True (previous behavior).
