# DICRAOS
DICAROS (Diffeomorphic Independent Contrasts for Ancestral Reconstruction of Shapes)

DOI: tbd

## Examples

Here are some examples of evolutionary trajectories reconstructed using DICAROS:

<div style="display: flex; justify-content: center;">
    <img src="Trajectory_Examples\Battus_belus_tree.gif" alt="Evolutionary Trajectory 1" style="margin-right: 10px;">
    <img src="Trajectory_Examples\Battus_belusbutterfly.gif" alt="Evolutionary Trajectory 2">
</div>

## Introduction 

This repository contains the code for the paper "DICAROS: Diffeomorphic Independent Contrasts for Ancestral Reconstruction of Shapes".
DICAROS is a method for reconstructing the ancestral shapes of a set of species using a set of leaf images. It is based off the Large Deformation Diffeomorphic Metric Mapping (LDDMM) to model smooth, invertible transformations between shapes while preserving the relationships between landmarks with Felsenstein's Independent Contrasts (IC) to iteratively reconstruct ancestral shapes along the branches of a phylogenetic tree. 

## Dependencies

- **Jax Geometry**: A library for differential geometry computations
  - Repository: [jaxgeometry](https://github.com/ComputationalEvolutionaryMorphometry/jaxgeometry)
  - Version: 0.9.4

- **Hyperiax**: A library for differential geometry computations
  - Repository: [hyperiax](https://github.com/ComputationalEvolutionaryMorphometry/hyperiax)
  - Version: 1.0.1
  - For this project, we used the version within this folder 

## Installation

1. Create and activate a conda environment:

   ```bash
   conda create -n DICAROS python=3.13
   conda activate DICAROS
   ```

2. Install required packages:

   ```bash
   pip install jaxdifferentialgeometry==0.9.4
   pip install pandas==2.2.3
   pip install HeapDict==1.0.1
   ```
## Usage 

See example in the `PhyloMorphoSpace.ipynb` file, to do root reconstruction and plot the PhyloMorphoSpace.

For the evolutioanry trajectory, see the `image_evol.ipynb` file, to do the root reconstruction and plot the evolutionary trajectory for a leaf image and lift it to the root. 
Output examples are shown in the `output_examples` folder. 

## Citation 

DOI: tbd

## Contact
If you experience problems or have technical questions, please contact [Michael Severinsen](mailto:michael@mail-lind.dk)
