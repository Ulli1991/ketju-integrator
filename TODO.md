TODO
====

- Also other stuff marked in the code

Correctness
-----------

% Divide and conquer prim broken for small systems (2 particles)
    + force using standard prim in this case
    - should be fixed properly

- `safetybuf` in allocate_internal_state seems worrying 

Optimization
------------
- The divide and conquer code has a lot of dynamic allocation, which might be
  avoidable. May not be significant for performance, though.

- Investigate how the code vectorizes, particularly force calculation and gbs
  stuff. Best effect likely with -Ofast, since vectorized sqrt usually not enabled without it.

- Currently the timers do not capture enough things, for large task numbers some
  30% of the time is missed from the subcategories

Features
--------

- Add spin terms
- BH mergers

Style
-----
- Clean up the internal implementation naming etc. Some parts of the code need
  more drastic reorganization as well

- Make all internal functions static/inline to avoid polluting the namespace,
  make sure all other symbols have the ketju prefix

Other
-----

- Improve the standalone integrator

- Python bindings

- Automated tests

- Automated benchmarks
