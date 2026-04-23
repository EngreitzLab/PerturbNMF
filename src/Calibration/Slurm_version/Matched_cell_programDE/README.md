# Matched cell programDE

## Overview
This R pipeline performs statistical testing to identify significant perturbation effects on gene programs using propensity score matching and regression analysis. environment file is provided to install necessary packages. Written by Tony Zeng. 

## Pipeline Steps

### 1. Cell Matching with Propensity Score Model
Select matched cells (perturbed vs unperturbed) using propensity score matching:

```
logit(πc) = log(Pr(Tc = 1|Xc) / (1 - Pr(Tc = 1|Xc))) = γ⊤Xc
```

### 2. Program Score Transformation
Apply log transformation with adaptive pseudocount to program scores:

```
Ỹck = log(Pck + ψk)
```

### 3. Regression with G-computation
Using matched cells, transformed program scores, and selected covariates, perform regression:
```
Ỹck = β0 + βgTc + β⊤XcXc + ϵck
```

where `Tc` is the treatment indicator (1 for perturbed, 0 for unperturbed).

**G-computation:**
- Estimate `beta_effect` by computing the difference in predicted outcomes:
  - `Ŷ_treatment`: predicted value when setting Tc = 1 for all cells
  - `Ŷ_control`: predicted value when setting Tc = 0 for all cells
  - `beta_effect = mean(Ŷ_treatment) - mean(Ŷ_control)`

### 4. Variance Estimation
Use delta method to calculate variance and standard deviation of the effect size.

### 5. Statistical Testing
Compare `(beta_effect, std)` to normal distribution using two-sided test to obtain p-values.

## Null Model Testing

For the null model, group 6 guides into 1 pseudo gene and follow the same process:
1. Match cells
2. Transform program scores  
3. Regression with g-computation → `beta_effect` and `std`
4. Two-sided test against normal distribution → p-values