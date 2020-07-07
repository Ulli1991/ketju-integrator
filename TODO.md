TODO
====

- Also other stuff marked in the code

Correctness
-----------

% Divide and conquer prim broken at least for small systems (2 particles) and
  for small particle/task counts
    ? Maybe fixed now?
    + force using standard prim in the small particle count case
    - Standard prim might be good enough for ~500-1000 particles
    ? should be fixed properly (if still broken),
      apparently can break down even with larger systems.

- Softening with the gadget kernel appears to be numerically unstable when
  combined with the high-order integrator used here. 
  Probably best to remove the option completely.

Optimization
------------
- The divide and conquer code has a lot of dynamic allocation, which might be
  avoidable. May not be significant for performance, though.

- Currently the timers do not capture enough things, for large task numbers some
  30% of the time is missed from the subcategories

Features
--------

% BH mergers
    + Basic merging conserving momentum, mass
    - GR effects on mass loss, resulting spin, kicks

Style
-----
% Clean up the internal implementation naming etc. Some parts of the code need
  more drastic reorganization as well
  - Mainly divide and conquer code left to do.

+ Make all internal functions static/inline to avoid polluting the namespace,
  make sure all other symbols have the ketju prefix

Other
-----

- Improve the standalone integrator

+ Python bindings
    + Mostly work
    - Could be easier to use, but maybe simple is good enough
    - Some things are probably not portable enough
    - expose the performance counters

+ Automated tests
    + Basics setup and some tests done
    - More comprehensive testing

- Automated benchmarks
