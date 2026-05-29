# An Educational From-Scratch Hartree–Fock Implementation in C++/CUDA

## Motivation
This project was created to improve my understanding and practical experience with computational methods for solving the Schrödinger equation as a second-year Physics student at Imperial College London.

## Current Status

### UHF development

I am currently working on the final planned technical upgrade: the addition of a `uhf_solver` function in `scf.hpp`. This implementation will use separate spin-up and spin-down density matrices, enabling the simulation of open-shell systems. On-the-fly Fock matrix construction is not currently planned, as the available system memory is sufficient for ERI tensors whose computation would already require prohibitively long wall times (typically exceeding 24 hours)..

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
## Computation of two-electron integrals on different hardwares
### GPU acceleration benchmarking
Both the CPU and GPU implementations have been validated on a variety of molecular systems using basis sets up to cc-pVTZ. The two implementations produce identical results for all tested molecules, and the computed energies are consistent with literature values and close to the Hartree–Fock limit. 

**Hardware:** Intel i9-14900K, NVIDIA RTX 5090, Ubuntu 24.04.4 LTS  
**Basis Set:** cc-pVTZ  
**Benchmark:** Full RHF SCF calculations including ERI construction, Fock matrix construction, diagonalization, and density matrix updates.
| Molecule | Basis Size | CPU Time (s) | GPU Time (s) | Speedup |
|:-|:-:|:-:|:-:|-:|
| $\mathrm{CH}_4$ | 95 | 352 | 134 | X 2.63 |
| $\mathrm{Cl}_2$ | 78 | 2301 | 251 | X 9.17 |
| $\mathrm{MgO}$ | 74 | 1185 | 161 | X 7.36 |
| $\mathrm{CO}_2$ | 105 | 2018 | 520 | X 3.88 |
| $\mathrm{NH}_3$ | 80 | 207 | 76 | X 2.59 |
| $\mathrm{NaF}$ | 74 | 1180 | 159 | X 7.42 |
| $\mathrm{NaCl}$ | 78 | 2525 | 268 | X 9.42 |
| $\mathrm{SO}_2$ | 109 | 3883 | 609 | X 6.38 |

The reported timings correspond to complete SCF calculations rather than isolated ERI tensor construction. In practice, ERI evaluation dominates the computational cost for these systems, while all remaining SCF steps combined (Fock matrix assembly, matrix diagonalization, density matrix construction, convergence checks, and related operations) typically require only 0.5-5 seconds per calculation.

## Running CPU-only
If CUDA is unavailable, the code can be compiled and executed in CPU-only mode. To do so:

1. Remove lines such as

```cpp
#include "eri_gpu.hpp"
```

from `scf.hpp`.

2. Remove all calls to GPU-related functions such as `ERI_GPU` and `ERI_tensor` in `scf.hpp`.

3. Edit `CMakeLists.txt` to:
   - exclude all `.cu` source files from compilation,
   - remove the `find_package(CUDAToolkit REQUIRED)` dependency,
   - omit any CUDA-related target linking.

The resulting executable will use the CPU implementation exclusively.

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

GPU acceleration is available for 3D electron-density evaluation through `density3d_gpu`, which significantly improves performance by parallelizing evaluations over spatial grid points. `.cube` files can be used to plot isosurfaces in softwares such as VMD, PyMOL and Avogadro.

