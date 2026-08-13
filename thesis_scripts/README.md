# Thesis experiment code

This folder contains publication-named copies of the notebooks and helper scripts used for the thesis experiments and manuscript figures. The original working files are unchanged. Notebook outputs have been cleared to keep the release readable; experiment results are not duplicated here.

## 01_multidataset_experiment

- `cifar10_static_pca_benchmark.ipynb` and `patchcamelyon_static_pca_benchmark.ipynb` implement the main frozen-PCA benchmark reported under RQ1.
- Run the matching static notebook before its moving-window notebook.
- `cifar10_moving_window_pca_benchmark.ipynb` and `patchcamelyon_moving_window_pca_benchmark.ipynb` implement the fixed-reference moving-window comparison.
- `moving_window_pca_fixed_reference.py` is required by both moving-window notebooks.

## 02_observation_level_calibration_extension

- The two `*_cpv70_static_pca_base.ipynb` notebooks provide the 70%-CPV static pipelines loaded by the calibration-extension notebooks.
- `cifar10_observation_level_calibration.ipynb` and `patchcamelyon_observation_level_calibration.ipynb` resample IC observations before constructing overlapping windows and detector paths.
- `observation_level_sequential_calibration.py` implements the direct sequential calibration and held-out audit.
- `seed_paired_statistical_analysis.ipynb` and `seed_paired_method_analysis.py` provide the seed-level comparison utilities.

Run each CPV base notebook only when generating its standalone CPV benchmark. The observation-level notebook imports the base notebook with execution disabled and then performs its own calibration and evaluation.

## 03_figures_and_trace_diagnostics

These scripts generate the selected-configuration and complete static/moving-window score-path figures used for diagnostic illustration.

## 04_supporting_utilities

These scripts audit PCA retention and support the adaptive-PCA sensitivity work.

## Controlled CIFAR-10 experiment

The controlled memory, monitoring-level, PCA-dimension, and baseline experiment was supplied as a completed report (`tesis_experiments.pdf`). No matching source notebook or script was found in the thesis-results folder, Documents, or Downloads locations searched when this release was assembled. It is therefore not represented here as executable code.

## Data and environment

The notebooks expect local CIFAR-10 and PatchCamelyon data and may download pretrained model weights when they are not already cached. Output directories are created beneath each experiment folder. Paths and environment variables should be reviewed before a full run. Computationally intensive cells are disabled only where the notebook itself exposes a run flag.
