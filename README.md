# Generative scenario modelling for renewable-grid dispatch

SynGrid explores how generated wind and electricity-demand scenarios can inform grid dispatch. This repository accompanies **“Generative Scenario Modeling for Robust Dispatch in Renewable-Integrated Smart Grids,”** developed for the **IEEE IES Generative AI Challenge 2026** and presented at **IEEE IRAI 2026** in Melbourne. The authors are Sean Afamefuna, Sadia Rahman, Glory Simon, and Jessica Osunbor. The [conference poster](docs/IRAI2026_Presentation_Poster.pdf) is included below.

![IRAI 2026 presentation poster](figures/IRAI2026_poster_preview.png)

Wind generation and demand do not follow one fixed forecast. We trained a conditional diffusion model to generate paired, 24-hour wind and load scenarios from the preceding 48 hours of data. We compared those scenarios with simpler baselines, checked them against empirical feasibility limits, and used the generated **load** trajectories in optimal power flow (OPF) tests on the IEEE 30-bus case.

The poster presents a later, larger dispatch study: **24,000 OPF runs with a 59.19% success rate**. The dispatch scripts and results here cover an earlier **480-run experiment with a 93.54% success rate**. They are different experiments; the larger poster result cannot be reproduced from the files in this repository.

![Twenty generated load scenarios](figures/diffusion_20_scenarios_load.png)

## Repository contents

| Folder | Contents |
| --- | --- |
| `src/` | Python data preparation, diffusion model, baseline comparisons, sampling, metrics, feasibility checks, and plotting |
| `dispatch/` | MATLAB/MATPOWER scripts for IEEE 30-bus OPF experiments |
| `data/processed/` | Cleaned hourly German wind/load series, chronological CSV splits, and saved normalization statistics |
| `checkpoints/` | Best saved normalized diffusion checkpoint |
| `results/` | Numerical results, scenario arrays for selected samples, and table exports |
| `figures/` | Scenario and dispatch figures |
| `docs/` | IRAI 2026 presentation poster and project notes |

## Method and results

The data pipeline uses hourly German wind-generation and electricity-load observations from the OPSD dataset. Each example has 48 hours of history and a 24-hour target; target channels are **wind, then load**. The model generates 20 joint trajectories for each evaluated window. We evaluated **50 validation and 50 test windows**, with 20 scenarios per window. Lower scores are better.

| Test-set model | Wind CRPS | Load CRPS | Joint energy score |
| --- | ---: | ---: | ---: |
| Diffusion | 5,231.56 | **1,974.40** | **32,432.48** |
| Linear regression | **5,015.03** | 3,189.96 | 36,190.77 |
| Persistence | 5,804.28 | 9,149.97 | 65,116.32 |
| Historical sampling | 7,441.67 | 5,461.63 | 51,456.35 |

The diffusion model has the lowest load CRPS and joint energy score in this comparison; linear regression has the lowest wind CRPS. The [main comparison](results/probabilistic_metrics_comparison_50.csv) and [historical-sampling comparison](results/historical_stochastic_baseline_comparison_50.csv) contain the underlying results.

We then clipped negative wind/load values and hourly changes above the 99th-percentile absolute ramps measured in the training targets. Across the 50 test windows, this removed **232 negative wind entries and 2,597 wind-ramp violations**. The test energy score changed from **32,432.48 to 32,293.74**. These empirical checks do not establish full physical or network feasibility. The [feasibility comparison](results/ramp_feasible_probabilistic_metrics_comparison_50.csv) gives the results for both splits.

For the dispatch experiment, we converted generated **load** scenarios into uniform load-scaling factors for MATPOWER's IEEE 30-bus case. Of **480 independent OPF runs** (20 scenarios × 24 hours), **449 succeeded and 31 failed**. The **93.54% OPF success rate** describes convergence in this setup; it is not a forecast-accuracy score or a reliability guarantee. The [dispatch summary](results/case30_test_sample0_24h_dispatch_summary.csv) records each run.

![OPF success across scenarios and hours](figures/case30_24h_dispatch_success_heatmap.png)

## Running the code

Use Python with NumPy, pandas, PyTorch, SciPy, scikit-learn, and Matplotlib. Exact package versions are not pinned. Run these data-preparation scripts from the repository root:

```bash
python src/build_supervised_windows.py
python src/split_supervised_windows.py
python src/diffusion_data_prep.py
python src/normalize_diffusion_data.py
```

The included hourly and split CSVs let these scripts rebuild the NumPy training arrays. To start from the original OPSD `time_series_60min_singleindex.csv`, place that file in `data/raw/` and first run `src/prepare_opsd_subset.py`. The raw OPSD file is not included here. Sampling and evaluation need the rebuilt arrays and `checkpoints/diffusion_model_normalized_epoch_best.pt`. To train a new model after preparation, run `python src/train_diffusion_normalized_long.py`; training requires substantially more compute.

The scripts also include intermediate experiments, so these steps do not reproduce every figure and table in one command. For example, `python src/plot_case30_24h_dispatch_success_heatmap.py` regenerates the OPF heatmap from the included results CSV. Load the saved PyTorch checkpoint only in a trusted environment because checkpoint deserialization can execute code.

The MATLAB scripts use **MATPOWER 8.1** on the MATLAB path. Copy `results/case30_test_sample0_24h_load_scales.csv` into MATLAB's working directory before running `dispatch/run_case30_24h_scenario_load_scales.m`. The script uses MATPOWER's `case30` and `runopf` and writes its outputs to that working directory.

## Scope and limitations

- The IEEE 30-bus experiment scales bus demand uniformly and solves each hour independently. It does **not** inject generated wind into the buses, schedule reserves, enforce capacity ceilings, or solve a coupled multi-period OPF.
- The feasibility adjustment enforces non-negativity and empirical ramp limits. It is not a complete physical or network-feasibility model.
- The large raw OPSD CSV and generated NumPy training arrays are omitted. The included processed hourly series is sufficient to rebuild the arrays with the listed scripts, but the unbundled raw file is needed to repeat the raw-data cleaning step.
- The conference poster documents the larger 24,000-run study, but its outputs and reproduction code are not included here. A manuscript and formal publication link are also not available in this repository.

The repository does not currently specify an open-source licence.
