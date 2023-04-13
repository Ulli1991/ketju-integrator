TODO
====

- Also other stuff marked in the code

Features
--------
? Replace uses of terminate() with returned error codes?
  Would allow better handling of failures by the calling code, e.g. logging or
  even rewinding state instead of the current instant death.
  But would then require that users actually check this code always, and might
  require large changes to the code to propagate error codes from all the functions.

Optimization
------------
- The divide and conquer code has a lot of dynamic allocation, which might be
  avoidable. May not be significant for performance, though.

- `compute_T` might benefit from parallelization, but not clear whether comms
  overhead too big for this to be useful. If implemented, needs careful benchmarking. 

- `get_gbs_rows_for_groups` could account for uneven group sizes for more even
  compuational load. But generally probably a fairly small effect.

? Currently the timers do not capture enough things, for large task numbers some
  30% of the time is missed from the subcategories

Other
-----

+ Automated tests
    + Basics setup and some tests done
    ? More comprehensive testing

- Automated benchmarks
