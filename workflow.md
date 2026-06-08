## ProteinArea Workflow

## Overview

# ProteinArea computes the cross-sectional area of a protein along the normal axis of a membrane, A(z). Starting with preprocessed molecular dynamics trajectory, the workflow slices the protein along the z-axis, computes the cross-sectional area within each slice, and then generates an area profile that can be averaged across the trajectory.


## Input

# The script ProteinArea requires:

# * A structural file (.tpr)
# * A trajectory file (.xtc)
# * Lower and uppper z-bounds (zmin, zmax)
# * Slice thickness (dz)

## Running ProteinArea

# Example slurm file

# python ProteinArea_V3.py \
  step7_production.tpr step7_production_centered_wt2_stride10.xtc \
  --output "${output}" \
  --zmin 27 \
  --zmax 137 \
  --layer 2.5 \
  --backend multiprocessing \
  --workers "${SLURM_CPUS_PER_TASK}"


## Output

# The primary output is a NumPy array:

# * area_profile.npy

# The array dimensions are:

# * (number of frames, number of z-slices)

# Rows correspond to trajectory frames and columns correspond to z-slices.


## Workflow Summary

# 1. Preprocess the trajectory.
# 2. Select the z-window and slice thickness.
# 3. Run ProteinArea.
# 4. Save the generated area array.
# 5. Average over frames. 



