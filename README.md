k-DOP axis optimizers for "k-DOP Clipping: Robust Ghosting Mitigation in Temporal Antialiasing"
================================================================================

This repo contains two tools for precalculating optimized k-DOP axis sets, as used
in the paper [k-DOP Clipping: Robust Ghosting Mitigation in Temporal Antialiasing](https://doi.org/10.1145/3681758.3697996)
(to appear in SIGGRAPH Asia 2024 Technical Communications).

## Building

In addition to the standard library, the optimizers only depend on GLM.
Additionally, OpenMP is supported to speed up the brute-force optimization
process.

The programs have only been tested on Ubuntu 22.04, but due to the simple
dependencies, they should work fine anywhere.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Optimization logic

The optimizers try to select axes such that the bounding volume is minimized,
given optimizer-specific constraints.

We initially tried to use plain simulated annealing, which does kind of work, but
settled on a slight variation that randomly perturbs axes, and shrinks step size
if there has been no improvement for a set number of steps. This seemed to
require less tweaking to converge to an acceptable solution for this
optimization problem.

## Sphere optimizer

This optimizer does not assume any specific type of scene; it simply optimizes
axes to bound a sphere as tightly as possible. For this process, the extents
for each range are forced to [-1, 1].

Axis sets generated this way are pretty safe, in that they don't assume
anything about the scene.

```sh
build/sphere_optimizer <axis-count> [forced axes...]
```

For example, to replicate the 32-DOP variant used in the paper:

```sh
build/sphere_optimizer 16 1 0 0 0 1 0 0 0 1
```

This generates 16 axes (the k in k-DOP is always double the number of axes), the
first three of which are forced to be (1,0,0), (0,1,0) and (0,0,1). The rest of
the axes are optimized with knowledge of the forced axes.

**NOTE**: For replicating the exact same numbers as in our supplemental
material, you'll need to uncomment the CGAL volume calculation variant in
`sphere_optimizer.cc`. Our own volume solver is faster but also less precise,
causing slight differences in the result.

## Image optimizer

This optimizer minimizes the k-DOP volume around 3x3 color neighborhoods sampled
from a (preferably aliased) input image; it can be used to minimize the types
of ghosting that are likely prevalent in the given image. In the paper, we
found that compared to sphere-optimized axes, image-optimized sets can provide
similar quality with fewer axes. This means less performance overhead. However,
they may be less robust, especially if the scene does not match the image
used for optimization.

```sh
build/image_optimizer <image-path> <axis-count> [forced axes]
```

For example, to generate a 16-DOP with a specific input image:

```sh
build/image_optimizer path-to-image.png 8
```

This generates 8 axes such that the average 3x3 color neighborhood in that image
is bounded as tightly as possible.

As with the sphere optimizer, you can also define forced axes. Putting the X, Y
and Z axes there ensures that you get no more ghosting than RGB AABB clipping.

The image optimizer is non-deterministic when the OpenMP acceleration is
enabled, you may get different sets each run. This is due to a floating point
sum occurring in potentially different orders, causing rounding differences. As
such, it's near impossible to replicate the exact same numbers found in the
supplemental material, even if the same input images were to be used.

## Future work

These would be nice-to-have, but the authors don't currently plan on doing them:

* Allow image optimizer to optimize for multiple representative images at the same time
* Optimize k-DOP volume computation; [a recent paper](https://doi.org/10.1145/3681758.3697996)
  presents a faster way to evaluate the vertices of a k-DOP, which could be used for this.
* Make image optimizer fast enough to evaluate all 3x3 neighborhoods instead of sampling a subset
* Support EXR or some other high bit depth format for image optimizer

Pull requests are welcome!
