Changed
^^^^^^^

* Changed ``stupidSetup.sh`` to install the pinned Isaac Sim Python packages
  instead of requiring an Isaac Sim source checkout.
* Changed the installer to replace a mismatched Isaac Sim package with the
  pinned version.
* Ignored a top-level ``IsaacSim`` directory left by the former source
  submodule so publication checkouts remain clean after an update.
* Updated the publication Isaac Sim package pin to ``6.0.1.0``.
