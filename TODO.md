TODO
====

- Also other stuff marked in the code

Optimization
------------
- The divide and conquer code has a lot of dynamic allocation, which might be
  avoidable. May not be significant for performance, though.

- Currently the timers do not capture enough things, for large task numbers some
  30% of the time is missed from the subcategories

Other
-----

+ Python bindings
    + Mostly work
    - Could be easier to use, but maybe simple is good enough
    - Some things are probably not portable enough, mainly the mpirun stuff.

+ Automated tests
    + Basics setup and some tests done
    - More comprehensive testing

- Automated benchmarks
