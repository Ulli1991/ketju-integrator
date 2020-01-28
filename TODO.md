TODO
====

- Also other stuff marked in the code

Correctness
-----------

- Divide and conquer prim broken for small systems (2 particles)
    - Maybe force using standard prim in this case?


- Valgrind sees an error in ketju_check_relative_proximity_ND_2, should be fixed

- `safetybuf` in allocate_internal_state seems worrying 

Optimization
------------
- The divide and conquer code has a lot of dynamic allocation, which might be
  avoidable. May not be significant for performance, though.

- Investigate how the code vectorizes, particularly force calculation and gbs
  stuff. Best effect likely with -Ofast, since vectorized sqrt usually not enabled without it.

Features
--------

- Add spin terms, test that PN stuff works
- BH mergers

Style
-----
- Clean up the internal implementation naming etc. Some parts of the code need
  more drastic reorganization as well

- Make all internal functions static/inline to avoid polluting the namespace,
  make sure all other symbols have the ketju_ prefix

Other
-----

- Improve the standalone integrator

- Python bindings

- Automated tests

- Automated benchmarks
