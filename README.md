# CONCAVE
Convex Optimization for Numerical Computations Achieving Verifiable Estimates

## Installation

There are two ways to install and use this package. Both assume that Julia is
already present on the system.

Assuming that only the Julia package `CONCAVE` itself is desired, you can add
it to your Julia environment with
```
julia> Pkg.add("https://github.com/lanl/CONCAVE")
```

Alternatively, to download the full source in "development mode":
```
$ git clone https://github.com/lanl/CONCAVE
$ cd CONCAVE
$ julia
julia> Pkg.activate(".")
```

## LANL open-source release

This project is part of the suite "Convex methods for physics", known
internally as O#4725. It is released under the BSD-3 license (see `LICENSE`).

```
© 2024. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for
Los Alamos National Laboratory (LANL), which is operated by Triad National
Security, LLC for the U.S. Department of Energy/National Nuclear Security
Administration. All rights in the program are reserved by Triad National
Security, LLC, and the U.S. Department of Energy/National Nuclear Security
Administration. The Government is granted for itself and others acting on its
behalf a nonexclusive, paid-up, irrevocable worldwide license in this material
to reproduce, prepare. derivative works, distribute copies to the public,
perform publicly and display publicly, and to permit others to do so.
```

