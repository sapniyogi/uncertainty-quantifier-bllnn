# bllnn

A Bayesian Last Layer neural network, exposed as a **composable
conditional-model block** for Gibbs samplers you write yourself -- the
neural-network analogue of what [`dbarts`](https://CRAN.R-project.org/package=dbarts)
does for Bayesian Additive Regression Trees.

Hand it a residual and a noise level; get back one honest posterior draw of a
nonlinear function. That contract is the whole package.

> **Status:** working and tested, not yet on CRAN. 1229 tests, `R CMD check`
> clean. The API is stable enough to build on, but not yet frozen.

## What this is, and what it is not

**It is a component.** You write the sampler; this supplies one step of it. If
you have a model where some nonlinear function needs to be drawn from its
conditional posterior each sweep, this is that draw.

**It is not a neural network library,** and not primarily a `fit()`/`predict()`
estimator. A convenience wrapper exists, but it is a thin layer over the
sampler -- never the other way round. It is also not competitive with Gaussian
processes on predictive accuracy, and is not pitched that way; the claim is
composability and valid conditional draws.

The use case it was built for is partial-linear causal inference,

```
y = X b + f(Z) + e
```

where `b` needs an honest credible interval and `f(Z)` has to absorb nonlinear
confounding without stealing from `b`.

## Installation

```r
# install.packages("remotes")
remotes::install_github("sapniyogi/uncertainty-quantifier-bllnn")
```

Requires a working C++ toolchain (Rtools on Windows).

## The thirty-second version

```r
library(bllnn)

sim <- sim_partial_linear(n = 400, beta = c(treat = 1.5), p_z = 5,
                          confounding = 0.6, seed = 1)

fit <- bllnn(y ~ z1 + z2 + z3 + z4 + z5, data = sim$data,
             linear = ~ treat, seed = 1)

summary(fit)
```

```
Linear coefficients (95% credible intervals):

      Estimate     SD  Lower  Upper ESS
treat   1.4757 0.0955 1.2947 1.6591 638 *

* interval excludes zero
```

The true coefficient is 1.5, and the interval covers it. Ordinary least
squares on the same data returns **1.865** -- the confounding in `f(Z)` lands
squarely on the coefficient you care about.

## The part that matters: the block

The wrapper above is a convenience. The reason this package exists is that you
can drop the draw into a sampler of your own, exactly as you would `dbarts`:

```r
cf   <- bllnn_crossfit(z, y, linear = x, folds = 5)   # learn features once
mod  <- bllnn_sampler(cf, tau2 = "auto")
xr   <- partial_out(cf)                               # x - E[x | z]

stopifnot(is_valid_kernel(mod))

for (s in 1:n_iter) {
  # ... your draw of b given f and sigma ...
  set_response(mod, y - as.vector(xr %*% b))  # the residual this block explains
  set_sigma(mod, sigma)                       # you own the noise scale, always
  f <- gibbs_step(mod)                        # one exact posterior draw of f
  # ... your draw of sigma given b and f ...
}
```

Three properties make that loop legitimate rather than merely convenient:

1. **The draw is exact, not approximate.** With frozen features the conditional
   posterior of the last layer is exactly Gaussian, and it is drawn in closed
   form -- no inner MCMC, no variational step.
2. **You own `sigma`.** The block never estimates the noise variance. Doing so
   internally would use the data twice and break the joint chain.
3. **Invalid kernels are refused.** Not every posterior/feature combination is
   a valid Gibbs transition. `gibbs_step()` errors unless
   `is_valid_kernel(mod)` is `TRUE`, overridable only with an explicit
   `force = TRUE`. Silently producing wrong inference is the worst failure
   mode available here, so it is made loud.

## Outcome families

| `posterior =` | Valid Gibbs kernel | Use for |
|---|---|---|
| `"conjugate"` | yes, exact | Gaussian outcomes. The default |
| `"polyagamma"` | yes, exact | Binary outcomes, and negative-binomial counts via `dispersion` |
| `"laplace"` | **no** -- approximate | Poisson counts, which the Polya-Gamma identity cannot reach, and post-hoc uncertainty on a body trained outside any sampler |

`"laplace"` is not a kernel and `gibbs_step()` refuses it without
`force = TRUE`. For a Gaussian outcome it coincides with the exact conjugate
draw to machine precision; for others it is a genuine approximation, measured
in `inst/validation/laplace_checks.R`.

## Why the intervals are honest

Two design decisions do the work, and both are non-negotiable:

**Features are frozen.** The network body is learned once on held-out folds and
then fixed. This is sample splitting, not an optimisation -- it is what makes
the conjugate draw exactly correct conditionally.

**The linear terms are residualised against the nuisance.** Supply the linear
design to `bllnn_crossfit(linear = x)` and it estimates `E[x | z]`, then hand
the host `partial_out(cf)` in place of the raw design. This is Robinson
partialling-out, the identifying step behind double machine learning. Without
it the network can represent `E[x | z]` exactly and competes with `b` for the
same direction, attenuating it.

Over 100 simulated datasets at confounding 0.6, nominal 95% intervals covered
the true coefficient **96%** of the time, with bias **-0.020** where naive
least squares carried **+0.397**. The script is
`inst/validation/coverage_simulation.R`.

## Known limitations

Stated plainly, because they are real.

- **A Gaussian last layer gives a Gaussian predictive distribution.** It cannot
  represent skewed or multimodal conditionals.
- **Pointwise credible bands for `f(z)` undercover.** They quantify uncertainty
  about the best fit *within the span of the frozen features*, and say nothing
  about the distance from that span to the truth. On a smooth test problem,
  nominal 95% bands covered `f` at 58%. This is a property of the frozen-feature
  design and not of any particular posterior -- the exact conjugate path and the
  Laplace path give the same answer. The coverage claim above is about `b`,
  which is a different quantity protected by a different argument.
- **Not a predictive-accuracy story.** BLL variants generally do not beat
  Gaussian processes on standard regression benchmarks.
- **Negative-binomial dispersion is fixed, not estimated.** It must be a
  positive integer; profile over a few values.

## Roadmap

Vignettes and a worked demonstration on a public dataset; then CRAN
preparation; then a C++ port of the numeric core, with the plain-R
implementation kept permanently as the reference the port is tested against.

## References

- Polson, Scott & Windle (2013), Bayesian inference for logistic models using
  Polya-Gamma latent variables. *JASA* 108(504), 1339-1349.
- Hahn, Murray & Carvalho (2020), Bayesian regression tree models for causal
  inference. *Bayesian Analysis* 15(3), 965-1056.
- Chernozhukov et al. (2018), Double/debiased machine learning. *The
  Econometrics Journal* 21(1), C1-C68.

## License

GPL-3.
