# Perturbation Association Calibration

This folder contains three statistical tests for evaluating perturbation effects on gene programs identified by cNMF. Each test provides different statistical frameworks for assessing whether perturbations significantly affect gene program activity.

## Overview of Tests

### 1. U-test Perturbation Calibration
**Location:** `JupterNote_Version/U-test_perturbation_calibration.ipynb` and `Slurm_version/U-test_perturbation_calibration/`

**Description:** Computes perturbation association tests on real and fake (calibration) data using Mann-Whitney U-test to evaluate the statistical properties of perturbation detection methods.

**Method:**
- Uses non-parametric Mann-Whitney U-test to compare program usage between perturbed and control cells
- Tests both real perturbations and synthetic calibration data (non-targeting guides)
- Evaluates false positive rates through QQ plots comparing observed vs. expected p-value distributions

**Key Features:**
- Non-parametric approach (no distributional assumptions)
- Calibration analysis to assess statistical validity
- Suitable for initial exploratory analysis

---

### 2. CRT (Conditional Randomization Test)
**Location:** `JupterNote_Version/ProgramDE.ipynb` and `Slurm_version/CRT/`

**Description:** Uses SCEPTRE-inspired methodology with conditional randomization tests (CRT) to test perturbation effects on gene programs while accounting for technical covariates.

**Method:**
- Employs conditional randomization tests (CRT) to generate null distributions
- Groups non-targeting controls (NTC) to create ensemble null models
- Performs permutation-based testing (default: 1023 permutations)
- Accounts for covariates (e.g., batch effects, cell cycle)

**Key Features:**
- Robust to covariates through permutation framework
- Controls for multiple testing through ensemble NTC groups
- Provides empirical null distributions
- More computationally intensive but statistically rigorous

**Parameters:**
- `--number_permutations`: Number of calibration iterations (default: 1024)
- `--number_guide`: Number of non-targeting guides per calibration iteration (default: 6)
- `--covariates`: Covariate keys in .obs to include as-is
- `--log_covariates`: Covariate keys in .obs to log1p-transform before inclusion
- `--guide_annotation_key`: Name(s) of non-targeting guide target labels (default: "non-targeting")
- `--FDR_method`: FDR correction method: "BH" or "StoreyQ" (default: "BH")

---

### 3. Matched Cell ProgramDE
**Location:** `Slurm_version/Matched_cell_programDE/`

**Description:** Performs statistical testing using propensity score matching and regression with G-computation to identify significant perturbation effects on gene programs. Written by Tony Zeng.

**Method:**

#### Pipeline Steps:

1. **Cell Matching with Propensity Score Model**
   ```
   logit(πc) = log(Pr(Tc = 1|Xc) / (1 - Pr(Tc = 1|Xc))) = γ⊤Xc
   ```
   Select matched cells (perturbed vs unperturbed) to reduce confounding.

2. **Program Score Transformation**
   ```
   Ỹck = log(Pck + ψk)
   ```
   Apply log transformation with adaptive pseudocount to program scores.

3. **Regression with G-computation**
   ```
   Ỹck = β0 + βgTc + β⊤XcXc + ϵck
   ```
   - `Tc`: treatment indicator (1 for perturbed, 0 for unperturbed)
   - G-computation estimates `beta_effect` by computing difference in predicted outcomes:
     - `Ŷ_treatment`: predicted value when setting Tc = 1 for all cells
     - `Ŷ_control`: predicted value when setting Tc = 0 for all cells
     - `beta_effect = mean(Ŷ_treatment) - mean(Ŷ_control)`

4. **Variance Estimation**
   Use delta method to calculate variance and standard deviation of effect size.

5. **Statistical Testing**
   Compare `(beta_effect, std)` to normal distribution using two-sided test to obtain p-values.

**Null Model Testing:**
For null model validation, group 6 guides into 1 pseudo gene and follow the same process to generate null p-value distributions.

**Key Features:**
- Causal inference framework using propensity score matching
- Accounts for confounding covariates through matching
- G-computation provides interpretable effect sizes
- Variance estimation through delta method
- Written in R with detailed documentation

---

## Folder Structure

```
Calibration/
├── README.md (this file)
├── JupterNote_Version/
│   ├── U-test_perturbation_calibration.ipynb
│   └── ProgramDE.ipynb
└── Slurm_version/
    ├── U-test_perturbation_calibration/
    │   ├── U-test_perturbation_calibration.py
    │   └── U-test_perturbation_calibration.sh
    ├── CRT/
    │   ├── CRT.py
    │   ├── CRT.sh
    │   └── README.md
    └── Matched_cell_programDE/
        ├── run_matching_de_batch.R
        ├── run.sh
        ├── environment.yml
        └── README.md
```

## Choosing a Test

**Use U-test when:**
- Performing initial exploratory analysis
- Need fast, non-parametric testing
- Want to assess overall calibration properties

**Use ProgramDE when:**
- Need robust covariate adjustment
- Want permutation-based statistical rigor
- Have sufficient computational resources for permutation tests

**Use Matched Cell ProgramDE when:**
- Strong confounding is suspected
- Want interpretable causal effect sizes
- Need propensity score matching framework
- Prefer regression-based approach

## Notes

- All tests require muon/scanpy objects with cNMF program usage scores
- Guide assignment and metadata must be properly formatted
- Non-targeting control guides are required for null model calibration
- Environment files are provided for each test where applicable
