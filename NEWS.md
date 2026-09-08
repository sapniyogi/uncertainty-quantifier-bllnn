# bllnn 0.1.0

First release.

`bllnn` exposes a Bayesian Last Layer neural network as a composable
conditional-model block for Gibbs samplers the user writes, in the spirit of
`dbarts`. Hand it a residual and a noise level; receive one posterior draw of
a nonlinear function.

## The sampler block

* `bllnn_sampler()` builds the block, with `set_response()`, `set_sigma()`,
  `set_offset()` and `gibbs_step()` driving it from a host loop. The object has
  reference semantics, so a host does not reassign it each sweep.
* `is_valid_kernel()` and `valid_kernels()` report which posterior and feature
  combinations are legitimate Gibbs transitions. `gibbs_step()` refuses the
  ones that are not unless given `force = TRUE`, because silently targeting the
  wrong distribution is the worst failure available here.
* The noise scale is always supplied by the caller and never estimated
  internally, which would use the data twice and break the joint chain.

## Outcome families

* `"conjugate"` draws the exact Gaussian conditional in closed form.
* `"polyagamma"` handles binary outcomes and negative-binomial counts by
  Polya-Gamma augmentation, exact at both steps. `rpolyagamma()` implements
  Devroye's alternating-series sampler, so nothing is truncated.
* `"laplace"` covers Poisson counts, which the augmentation identity cannot
  reach, and post-hoc uncertainty on a body trained outside any sampler. It is
  approximate and is not a valid kernel; `laplace_moments()` and
  `laplace_draw()` expose it directly.

## Features and the causal machinery

* `bllnn_warmup()` trains a network body of arbitrary depth, which is then
  frozen. Freezing is what makes the conditional draw exactly correct.
* `bllnn_crossfit()` does the same by K-fold cross-fitting, and `partial_out()`
  returns the linear design residualised against the nuisance -- Robinson
  partialling-out, the identifying step behind double machine learning.
* `bllnn()` assembles that configuration, with `print()`, `summary()`,
  `coef()`, `confint()`, `predict()` and `plot()` methods.
* `sim_partial_linear()` simulates from the model the package targets.

## Validation

Over 100 simulated datasets at confounding 0.6, nominal 95% credible intervals
covered the true coefficient 96% of the time, with bias -0.024 where naive
least squares carried +0.397. The script is
`inst/validation/coverage_simulation.R`, and it exercises the configuration the
defaults give you rather than a neighbouring one.

## Known limitations, documented rather than hidden

* Pointwise credible bands for `f(z)` undercover: they describe uncertainty
  about the best fit within the span of the frozen features and say nothing
  about the distance from that span to the truth. Measured at 58% against a
  nominal 95%. The coefficient interval is a different quantity and is not
  affected.
* A Gaussian last layer cannot represent skewed or multimodal conditionals.
* Negative-binomial dispersion is fixed rather than estimated, and must be a
  positive integer.
* This package is not competitive with Gaussian processes on predictive
  accuracy and is not pitched on it.
