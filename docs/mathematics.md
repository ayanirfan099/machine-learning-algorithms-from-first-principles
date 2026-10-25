# Mathematical guide
With an intercept column, predictions are $Xw$ and $n$ is the sample count.

## Linear regression
$L(w)=\|Xw-y\|_2^2/n$. Differentiating each squared residual gives
$\nabla L=2X^T(Xw-y)/n$; gradient descent updates $w\leftarrow w-\eta\nabla L$.
The Hessian is $2X^TX/n$: step sizes below $2/\lambda_{max}(2X^TX/n)$ converge for the quadratic (strictly convex when full column rank). Scaling changes conditioning. Cost per step is $O(nd)$; storage is $O(nd+d)$.

## Logistic regression
$p_i=\sigma(x_i^Tw)$, $\sigma(z)=1/(1+e^{-z})$.
$L=\sum_i[\log(1+e^{z_i})-y_i z_i]/n$.
Using $d\log(1+e^z)/dz=\sigma(z)$ yields $\nabla L=X^T(p-y)/n$.
`logaddexp` avoids overflow; sigmoid is computed using its log-domain identity. No regularization is applied. Separable data can drive weights toward infinity; fixed iteration budgets and probability agreement must be interpreted accordingly. Step cost $O(nd)$.

## KNN
Euclidean distance $\sqrt{\sum_j(x_j-z_j)^2}$ and Manhattan distance $\sum_j|x_j-z_j|$. Squared Euclidean preserves ordering. Majority vote breaks label ties by the sorted smallest label, and equal-distance neighbors preserve training order. Sorting costs $O(n\log n)$ per query plus $O(nd)$ distances; processing queries individually avoids a large query-by-training tensor.

## K-means
Minimize $J=\sum_i\|x_i-\mu_{c_i}\|^2$. Alternate nearest-centroid assignments and cluster means. Empty clusters are reinitialized to a sampled point; this may break monotonicity. Stop on centroid displacement tolerance or iteration budget. Select minimum final inertia across seeded restarts. Cost $O(rtnkd)$; this implementation allocates pairwise differences of size $O(nkd)$. The solution is local, not guaranteed globally optimal.

## PCA
Center $X_c=X-\bar X$. SVD gives $X_c=U\Sigma V^T$. Rows of $V^T$ define principal axes; explained sample variance is $s_j^2/(n-1)$. Transform $Z=X_cV_k$; reconstruct $\hat X=ZV_k^T+\bar X$. Sign is indeterminate. Ratio divides component variance by total variance; constant data returns zero ratios. Dense SVD costs approximately $O(nd\min(n,d))$. Euclidean reconstruction error decreases as components are added.

Central finite differences with step $10^{-6}$ test both analytical gradients independently, with absolute tolerance $10^{-7}$.
