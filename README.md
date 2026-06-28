# Enterprise Risk Assessment: Apple's Proposed Acquisition of Netflix

A Competitive Intelligence and Quantitative Risk Analytics Approach.

**Course:** ALY 6130, Enterprise Risk Management, Northeastern University
**Team:** Samuel Tweneboah-Kodua Nyamekye, Adebukola Fadina, Sandeep Kaur

## Overview

This project is an enterprise risk assessment of a hypothetical acquisition of Netflix by Apple Inc. It identifies the competitive risks of the deal, scores them qualitatively on a 1 to 9 scale, models the largest ones quantitatively, and recommends response strategies with key risk indicators.

Three competitive risks carry the analysis:

- **R1 Regulatory and antitrust** (score 63, High). Expected loss 12.8 billion dollars, P95 40.9 billion.
- **R2 Subscriber churn and competitive response** (score 56, High). Expected first year revenue at risk 1.9 billion, P95 4.4 billion.
- **R3 Cybersecurity and data integration** (score 40, Medium). Expected annual loss 0.20 billion, P99 1.41 billion.

The quantitative work covers probability distributions and expected loss, a machine learning churn model aligned with the key risk indicators, Monte Carlo simulation, and an integrated situation assessment that links the business and non business environment to the model inputs.

## Data

No real dataset exists for this hypothetical acquisition, so the analysis uses **synthetic data** that simulates the combined business. The generation process, parameters, and assumptions are documented in `build_analysis.py` and in `notebooks/quantitative_analysis.ipynb`. All draws use a fixed seed (42), so every figure and number reproduces exactly.

## Repository structure

```
enterprise-risk-project/
├── README.md
├── requirements.txt
├── build_analysis.py          # generates data, runs the models, saves figures
├── build_notebooks.py         # assembles the notebooks from the analysis
├── data/
│   ├── raw/                   # synthetic subscriber data, risk register
│   └── processed/             # processed data, simulation outputs, key_results.json
├── figures/                   # all charts used in the notebooks and report
├── notebooks/
│   ├── eda.ipynb
│   ├── qualitative_analysis.ipynb
│   ├── quantitative_analysis.ipynb
│   └── monte_carlo.ipynb
└── report/
    ├── final_report.pdf       # the full assessment
    └── ALY6130_Presentation.pptx
```
