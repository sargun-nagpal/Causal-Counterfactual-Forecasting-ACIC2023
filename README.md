# Causal Counterfactual Forecasting - ACIC 2023
This is my final project for the **Causal Inference** course (APSTA-GE 2012) taught by Prof. Jennifer Hill at NYU. I am participating in the annual [American Causal Inference Conference (ACIC) 2023](https://sci-info.org/annual-meeting/) data challenge.

## Project Intro/Objective

Supervised Machine Learning methods are typically used for time series forecasting but do not account for counterfactual outcomes and time-dependent confounders. These models only fit observed data. 

Observational Causal Inference methods, on the other hand, are typically used for estimating treatment effect estimands, and not for forecasting future potential outcomes. 

The problem statement is to predict future counterfactuals for 5 time steps under 6 different treatment assignments for every unit.

Findings and analysis: [Report]()

## Data
The data is simulated. However, drawing analogy to an e-commerce problem:

We have the sales (outcome) of ~4000 products for 95 weeks under one or more of 6 different pricing strategies (treatment levels). There are three binary and three numeric covariates, some of which are static product level features (example- Average sale, Product category), while others are temporal features (example- Inventory count).

The goal is to predict the sales of each item for the next 5 weeks under each pricing strategy (intervention).

## Estimand
The estimand is the RMSE error over all forecast time steps and counterfactual states for each unit.

```math 

    \sqrt{\frac{\sum_{i, k, w}(y_{i,k}(w) - \hat{y}_{i,k}(w))^2}{N * N_K * N_W }}

```
Where $N$ is the number of units, $N_K$ the number of time steps, $N_W$ the number of treatments, $i$ a unit, $k$ a time step, $w$ a treatment, $y$ the
ground truth outcome and $\hat{y}$ the predicted outcome.

## Results

## Methods

## Next Steps