# An Educational From-Scratch Hartree–Fock Implementation in C++/CUDA

## Motivation
This project was created to improve my understanding of, and practical experience with, computational methods for solving the Schrödinger equation as a second-year Physics student at Imperial College London.

## Current Status

I have completed this project, and it has exceeded my expectations. It runs cc-pVTZ reliably on both CPU and GPU. The RHF code gives quantitatively good results for most common molecules containing elements from the first three rows, achieving differences of less than 10 $\mu{\text{Ha}}$ compared with PySCF. However, the UHF code shows slight deviations from PySCF for some open-shell molecules, and RHF fails for a few organic molecules such as cytosine and benzothiophene, as well as molecules with CN triple bonds (cyanides). The failure for cyanides is due to numerical instability in the current Boys function implementation when handling integrals involving this type of bond.

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

Input is provided by passing a `.json` file on the command line when running the executable compiled from the application `.cpp` files.

```bash
./executable input.json 
```
`scripts/shell/tasks.sh` is an example of compiling an executable from a `.cpp` application using `cmake` and then running the executable with different JSON inputs on Linux.

The key names are hardcoded, and the executable expects essential JSON fields such as `"basis_path"`, `"save_path"`, and `"atoms"` to contain valid input. The JSON field `"atoms"` should contain sub-fields specifying the element symbol and position, either as a Cartesian position vector `[x, y, z]` in units of $a_0$ or as a spherical polar coordinate `[radius, polar, azimuthal]`, with the radius in $a_0$ and angles in degrees.

Here is an example JSON input for running a $\mathrm{NaCl}$ monomer on the CPU with the cc-pVTZ basis set.

```json
{
    "basis_path": "/home/jc6224/hf_cpp/basis",
    "save_path": "/home/jc6224/hf_cpp/data",
    "folder_name": "nacl_c",
    "basis": "cc-pVTZ",
    "charge": 0,
    "eri_hardware": "CPU",

    "atoms": [
        {
            "symbol": "Na",
            "coordinate_type": "cartesian",
            "position": [0.0, 0.0, 0.0]
        },
        {
            "symbol": "Cl",
            "coordinate_type": "cartesian",
            "position": [4.46, 0.0, 0.0]
        }
    ]
}
```
Different apps require different JSON keys, but they share most keys with the input of `single_run.cpp`. Example `.json` files for different apps are available in the `examples` folder.

### SCF Output
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

This is only the output structure for `single_run.cpp`; other apps may have slightly or completely different outputs.

---

## Applications

This repository includes four standalone applications built on top of the Hartree–Fock solver. They are compiled from the application `.cpp` files and are run by passing a `.json` input file on the command line.

### `single_run.cpp`

`apps/;single_run.cpp` performs one RHF or UHF calculation for a specified molecular geometry.

**Input:** a JSON file specifying the molecular geometry, basis set, basis path, output path, charge, SCF settings, ERI hardware (`"CPU"` or `"GPU"`), and, for UHF calculations, the alpha and beta spin occupations.

**Output:** a saved SCF result folder containing the molecular geometry, basis information, density matrix or spin density matrices, molecular orbital coefficients, orbital energies, metadata, timing information, and total energy.

This is the main executable for running a single Hartree–Fock calculation and generating data that can later be used by the visualization application.

### `bl_scan.cpp`

`apps/bl_scan.cpp` performs repeated RHF calculations over a user-defined set of bond lengths.

The molecule is specified using a central atom and a set of satellite atoms. Each satellite atom is assigned an angular position, and its distance from the central atom is varied during the scan. The user can also define several distance segments, each with its own start distance, end distance, and number of evenly spaced points. This makes the scan flexible enough for diatomic molecules and other systems with central symmetry.

**Input:** a JSON file specifying the central atom, satellite atoms, angular positions, basis set, charge, SCF settings, ERI hardware, and one or more distance segments.

**Output:** energies, HOMO and LUMO energies, SCF iteration counts, scan distances, timing information, and metadata suitable for plotting potential energy curves.

This application is useful for bond-length scans, dissociation curves, and simple radial geometry studies.

### `dissociation.cpp`

`apps/dissociation.cpp` computes the atomisation energy of a closed-shell molecule.

The molecular calculation is performed with RHF. The atomic reference calculations are performed separately for the isolated atoms, often using UHF because many individual atoms are open-shell systems. The final atomisation energy is computed from the difference between the sum of atomic energies and the molecular energy.

**Input:** a JSON file specifying a closed-shell molecular geometry, basis set, charge, SCF settings, and ERI hardware. Atomic spin occupations are assigned internally for supported atoms.

**Output:** the molecular RHF energy, individual atomic reference energies, the sum of atomic energies, the atomisation energy, timing information, and metadata.

This application currently supports closed-shell molecules only.

### `visualization.cpp`

`apps/visualization.cpp` evaluates electron densities and molecular orbitals from saved SCF data.

The user can choose either 2D or 3D evaluation. For 2D visualization, the input specifies a plane (`"xy"`, `"yz"`, or `"xz"`), a centre, side length, and resolution. For 3D visualization, the input specifies a cubic box by its centre, side length, and resolution. Both CPU and GPU grid evaluation are supported.

**Input:** a JSON file specifying the saved SCF data path, basis path, output path, visualization dimension, grid settings, render hardware (`"CPU"` or `"GPU"`), and which quantities to render. Supported quantities include the electron density, HOMO, LUMO, HOMO-1, LUMO+1, and the corresponding alpha/beta orbitals for UHF calculations.

**Output:** 2D `.csv` grid data for planar plots, 3D `.cube` files for volumetric visualization, and a `visualization_info.json` file recording the visualization settings and generated files.

This application is used to inspect electron densities and molecular orbitals after an SCF calculation.

---
## Computation of Two-Electron Integrals on Different Hardware
### GPU Acceleration benchmarking
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

## Running CPU-Only
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


## Algorithmic Optimizations
### Symmetry Exploitation
I implemented two important optimizations to speed up the evaluation of two-electron integrals. The first is symmetry exploitation. A two-electron integral

$$
\int \int  \frac{\chi_\mu(\mathbf{r}_1)\chi_\nu(\mathbf{r}_1)\chi_\lambda(\mathbf{r}_2)\chi_\sigma(\mathbf{r}_2)}{|\mathbf{r}_1 - \mathbf{r}_2|} d^3\mathbf{r}_1 d^3\mathbf{r}_2 = (\mu\nu|\lambda\sigma)
$$

where $\chi_i$ are basis functions. Using the commutative property of multiplication inside the integrand, we can find that

$$
(\mu\nu|\lambda\sigma) =(\nu\mu|\lambda\sigma) =(\mu\nu|\sigma\lambda)=(\nu\mu|\sigma\lambda).
$$

Also, the variables of integration $\mathbf{r}_1$ and $\mathbf{r}_2$ are interchangeable, therefore,

$$
=(\lambda\sigma|\mu\nu) =(\lambda\sigma|\nu\mu) =(\sigma\lambda|\mu\nu)=(\sigma\lambda|\nu\mu).
$$

This is the eightfold symmetry of the ERI tensor, which allows us to consider only the unique basis function quartets. An index mapping function in `indexing.hpp` enables the construction of Fock matrices directly from the array of unique ERI elements. This is completely equivalent to using the full ERI tensor.

The unique quartets are unique combinations of unique pairs of basis functions. For a basis size $N$, there are $N(N+1)/2$ unique pairs, and subsequently $\frac{1}{8}N^4 + \frac{1}{4}N^3 + \frac{3}{8}N^2 + \frac{1}{4}N$ unique quartets. This is much smaller than the full tensor, which has $N^4$ elements, significantly reducing computational cost and memory usage.

### Schwarz Screening

There is often some degree of sparsity in ERI tensors, which can be used to reduce computational cost. By using the Cauchy-Schwarz inequality, we can compute upper bounds for all two-electron integrals, allowing us to skip numerically insignificant ones by setting them directly to zero.

In Dirac notation, the Cauchy-Schwarz inequality can be written as
$$
| \langle m | n \rangle |^2 \leq \langle m|m \rangle\langle n|n \rangle,
$$
where $m$ and $n$ are two complex functions. It stays true in the presence of a positive semidefinite operator $\hat{O}$

$$
| \langle m |\hat{O}|n \rangle |^2 \leq \langle m|\hat{O}|m \rangle\langle n|\hat{O}|n \rangle.
$$

A two-electron integral 

$$
\int \int  \frac{\chi_\mu(\mathbf{r}_1)\chi_\nu(\mathbf{r}_1)\chi_\lambda(\mathbf{r}_2)\chi_\sigma(\mathbf{r}_2)}{|\mathbf{r}_1 - \mathbf{r}_2|}  d^3\mathbf{r}_1  d^3\mathbf{r}_2  = \int \int  \frac{\chi_\mu(\mathbf{r}_1)\chi_\lambda(\mathbf{r}_2)}{\sqrt{|\mathbf{r}_1 - \mathbf{r}_2|}} \frac{\chi_\nu(\mathbf{r}_1)\chi_\sigma(\mathbf{r}_2)}{\sqrt{|\mathbf{r}_1 - \mathbf{r}_2|}}d^3\mathbf{r}_1  d^3\mathbf{r}_2
$$

$$
= \langle \mu\lambda |\frac{1}{|\mathbf{r}_1 - \mathbf{r}_2|}| \nu\sigma \rangle
$$

$$
\leq \sqrt{\langle \mu\lambda |\frac{1}{|\mathbf{r}_1 - \mathbf{r}_2|}| \mu\lambda \rangle \langle \nu\sigma |{\frac{1}{|\mathbf{r}_1 - \mathbf{r}_2|}}| \nu\sigma \rangle} 
$$

$$
= \sqrt{(\mu\lambda|\mu\lambda)(\nu\sigma|\nu\sigma)}
$$

Hence, we can evaluate all the self-integrals of unique pairs of basis functions, $(\mu\lambda|\mu\lambda)$, of which there are $N(N+1)/2$. This gives upper bounds for all two-electron integrals and often allows many expensive recursions to be skipped.

### GPU Acceleration

Parallelization of two-electron integrals is key to making Hartree–Fock algorithms run faster. By assigning each CUDA block to a basis function quartet, and each thread within the block to a primitive Gaussian quartet, the ERI computation is parallelized at a useful level.

This implementation achieved a considerable performance gain compared with the CPU implementation, as shown above. However, there are limitations to using GPUs for scientific computations. Modern GPUs are designed mainly for AI workloads and excel at low-precision, simple floating-point operations. The McMurchie-Davidson recursion algorithm implemented here for one-electron and two-electron integrals reaches different recursion depths for Cartesian Gaussians with different angular momentum levels. As a result, some GPU threads finish earlier than others within a block, but they still need to synchronize and perform a local sum, which can leave many threads idle for long periods. During the execution of this code, the GPU draws less than half of its TDP. ERI computation could still be made much more GPU-friendly through further software optimization.


### DIIS Acceleration

Self-consistent field iterations can converge slowly or even fail when the density matrix oscillates. To improve convergence, I implemented Direct Inversion in the Iterative Subspace (DIIS), also known as Pulay mixing.

Instead of diagonalizing only the newest Fock matrix, DIIS builds an extrapolated Fock matrix from previous iterations,

$$
F_{\mathrm{DIIS}} = \sum_i c_i F_i.
$$

The coefficients are chosen to minimise the SCF error. For RHF, the error matrix at iteration i is

$$
R_i = F_i P_i S - S P_i F_i,
$$

where $F$, $P$ are Fock and density matrices of the current ($\text{i}_\text{th}$) iteration, and $S$ is the overlap matrix. The error matrix elements should vanish at self-consistency. The Pulay matrix is built from the inner products of stored error vectors,

$$
B_{ij} = R_i \cdot R_j,
$$

and the coefficients have contraint

$$
\sum_i c_i = 1.
$$

The coefficients $c$ are obtained by solving a matrix equation involving the augemented pulay matrix while enforcing the constraint.

For UHF, the alpha and beta Fock matrices and error vectors are combined so that one shared set of DIIS coefficients is used for both spin channels.

DIIS reduces the number of expensive SCF cycles for larger systems and can make calculations converge when simple damping alone would otherwise oscillate or fail.