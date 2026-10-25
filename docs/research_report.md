# Numerical comparison report
## Abstract
Five explicit numerical estimators were executed on controlled synthetic datasets. Linear prediction agreement with sklearn was 3.55e-15; both KNN metrics had exact label agreement on this run.
## Results
See the committed JSON and figures. Logistic probability differences reflect optimizer tolerances and finite-step optimization; sklearn uses a different optimizer. K-means initialization differs, so matching inertia here does not establish equivalence on arbitrary datasets. PCA signs are arbitrary; compare variances and subspaces rather than raw signs.
## Threats to validity
One seed, small synthetic datasets, no scalability benchmark and fixed hyperparameters. Test data were not used for tuning. Runtime is machine-specific. These experiments establish numerical correctness on representative cases, not broad predictive superiority.
## Conclusion
The executed checks support the educational implementations; broader robustness and real-world datasets remain future work.
