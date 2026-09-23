Fixed
^^^^^

* Fixed the Newton viewer raising a Dear ImGui "conflicting ID" warning caused by a
  duplicate "Show Inertia Boxes" checkbox. The Isaac Lab joint-limit and zero-reference
  toggles now render inside Newton's single "Visualization" header instead of a separate
  "IsaacLab Visualization" header, after Newton moved left-panel rendering from ``ViewerGL``
  into the shared ``ViewerGui``.
