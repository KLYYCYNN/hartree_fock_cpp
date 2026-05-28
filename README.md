# An Educational From-Scratch Hartree–Fock Implementation in C++/CUDA

## Motivation
This project was created to improve my understanding and practical experience with computational methods for solving the Schrödinger equation as a second-year Physics student at Imperial College London.

## Current Status

### CUDA Update
The GPU implementation is currently undergoing a major redesign. Previously, electron repulsion integral (ERI) computation was parallelized only at the basis-function level, which provided little to no acceleration compared to the CPU implementation on an Intel i9-14900K and NVIDIA RTX 5090 system.

I am currently working on parallelizing ERI computation down to the primitive Gaussian level. This approach is significantly more GPU-friendly, reduces both memory usage and computational stress on individual threads, and should lead to meaningful acceleration compared to the CPU implementation.


To disable GPU support and use a pure CPU implementation, remove:

```cpp
#include "eri_gpu.hpp"
```

and all calls to GPU-related functions such as `ERI_GPU` and `ERI_tensor` in `scf.hpp`. Also, edit `CMakeLists.txt` to omit the compilation of `.cu` files when building executables.

---

## I/O and Data Structure Design

### Basis Set Parameters
Basis set parameters are stored in `.csv` files, with separate files for each element and separate folders for different basis sets.

Each primitive Gaussian occupies one row in the file and stores parameters such as:
- contraction coefficient
- exponent

The `shell_id` column specifies shell membership for each primitive. The primitives within a shell are contracted to generate 1, 3, 6, or 10 basis functions depending on the angular momentum level `L`.

The main basis builder, `basis_construction` in `basis.hpp`, iterates over shells and appends the basis functions generated from each shell for every atom. It also stores the basis function information in a consistent order, allowing reconstruction for visualization after the SCF run completes.

### Molecular Input
At the current testing stage, molecular systems are defined using:

```cpp
std::vector<std::pair<std::string, vector3>>
```

which stores atom names and positions.

In future versions, direct editing of `.cpp` files will no longer be necessary. Molecular input will instead be handled through `.json` files using entries such as:

```json
"atoms"
"positions"
```

with additional support planned for automated bond-length scans of simple systems.

### Output Data
The functions in `save.hpp` handle output generation and storage of SCF results.

The function `save_scf_data` creates a folder at the specified `save_path` and stores files such as:

```text
basis.csv
density.csv
coefficient.csv
geometry.csv
orbital_energies.csv
metadata.json
```

These files allow full reconstruction of the basis set and wavefunction data for electron density analysis and molecular orbital visualization.

`metadata.json` also stores timing information for benchmarking purposes.

A separate output structure for multiple SCF runs (e.g. bond-length scans) is planned for future development.

---

## Visualization

A complete pipeline for visualizing electron densities and molecular orbitals (e.g. HOMO/LUMO) in both 2D and 3D is implemented in `cube.hpp`.

### 2D Visualization
For 2D visualization, an arbitrary square plane in 3D space can be specified using:
- the center position
- two side-direction vectors

The resulting data is stored as a `.csv` file with a 2D structure that can easily be loaded into a `numpy.ndarray` in Python and visualized using:

```python
matplotlib.pyplot.imshow
```

### 3D Visualization
For 3D visualization, electron density or molecular orbitals can be evaluated inside a cubic volume specified by:
- center position
- side length
- spatial resolution

A `.cube` file can then be generated using:
- the `cube_data` output type
- the function `write_cube_file`

GPU acceleration is available for 3D electron-density evaluation through `density3d_gpu`, which significantly improves performance by parallelizing evaluations over spatial grid points.

---

## Goals
The current codebase is a partially optimized CPU implementation of Restricted Hartree–Fock (RHF).

Existing optimizations include:
- matrix symmetry exploitation
- ERI tensor symmetry
- Schwarz screening for ERI evaluation

Future goals include:
- completing primitive-level GPU parallelization for ERI computation
- achieving significant GPU acceleration over the CPU implementation
- extending `scf.hpp` and related data structures to support Unrestricted Hartree–Fock (UHF)
- supporting open-shell systems such as $\mathrm{O}_2$