Fixed
^^^^^

* Fixed PhysX-backed Newton visualization shadows applying collision mesh
  approximations even though they only consume visual geometry and synchronized
  body poses, while keeping collider-only links visible in the Newton viewer.
