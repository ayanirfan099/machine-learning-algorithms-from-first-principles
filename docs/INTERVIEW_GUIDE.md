# Study guide

These notes explain implemented methods; do not claim independent authorship or experiments beyond the recorded results.

## Why scale features before gradient descent?

Scaling improves conditioning of the quadratic loss and permits a useful shared step size; it must be fit on training data only.

## Why can logistic weights grow on separable data?

Unregularized likelihood approaches its optimum as separating coefficients increase; finite-step convergence is not a finite maximum-likelihood solution.

## Why compare PCA subspaces rather than signed eigenvectors?

Singular-vector signs and bases within repeated-eigenvalue subspaces are not uniquely identified; reconstruction and projection matrices are more meaningful.
