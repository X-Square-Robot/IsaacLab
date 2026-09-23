Fixed
^^^^^

* Fixed ``ImportError: cannot import name 'warn' from 'warp._src.utils'`` when running
  PhysX (Kit) workflows with Newton actuators: the ``omni.warp.core`` extension ships
  warp at its extension root (outside any ``pip_prebundle`` directory), so the install
  prebundle repoint never covered it and Kit mixed the bundled warp with the
  environment's newer copy. The repoint scan now also covers ``omni.warp.core``
  extension roots, and ``newton_usd_schemas`` was added to the repoint
  package list so stale prebundled copies no longer shadow the environment packages.
