Fixed
^^^^^

* Fixed the Pink IK dependency force-install to pin ``eigenpy`` and ``coal``
  alongside ``pin``/``pin-pink`` so the whole cmeel/boost generation is pulled in
  one consistent step. The shared ``cmeel.prefix`` directory holds only one
  ``libboost`` version, so reinstalling ``pin`` alone left ``coal_pywrap.so``
  linked against a boost the directory no longer provided
  (``ImportError: libboost_serialization.so.<ver>``), breaking every
  ``pinocchio`` import.
