Added
^^^^^

* Added :obj:`CX002_FIXED_BASE_CFG` for fixed-root MJWarp position-control
  smoke tests that preserve full collision geometry.

Changed
^^^^^^^

* Changed the CX002 and EX001-family MJWarp demos to keep USD-authored
  CoACD startup generation under the default importer behavior without a
  demo-level enable/disable CLI override.

Fixed
^^^^^

* Fixed the :obj:`CX002_CFG` smoke-test defaults to use an explicit home pose
  and grouped actuator gains while keeping self-collisions enabled by default.
* Fixed the CX002 MJWarp demo to keep ``--collision_simplification none``
  usable with a safe contact preset and adjacent-link self-collision filters.
