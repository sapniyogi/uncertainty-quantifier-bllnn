# bllnn

**Honest credible intervals for a linear coefficient when the nuisance function
is nonlinear and you have no theory about its shape.**

`bllnn` fits the partial linear model

```
y = x'b + f(z) + e
```

by putting a neural network on `f(z)` and drawing `b` from a Gibbs sampler that
treats the network's last layer as a conjugate Bayesian regression. The target
is `b` and an interval for it that you can defend. The network is machinery.

It is also, deliberately, a **component**. The draw of `f` is exposed as a
single Gibbs step you can call from a sampler you wrote yourself, the way
[`dbarts`](https://CRAN.R-project.org/package=dbarts) exposes BART. If your
model has more in it than a partial linear mean -- random effects, a
measurement-error layer, a selection model, missing data -- you keep your
sampler and use this for one conditional.

> **Status:** working and tested; not yet on CRAN. 1242 tests, `R CMD check`
> clean. The API is stable enough to build on but is not frozen.

## The problem

You want the coefficient on a treatment, exposure or policy variable. You also
have covariates that influence both it and the outcome, through a relationship
you have no reason to believe is linear.

Two standard approaches both fail, and they fail in opposite directions.

**Force it into a linear model.** Whatever curvature `f(z)` has that the linear
terms cannot absorb is pushed into the coefficients, and it lands hardest on
whichever covariate is best positioned to absorb it -- often the treatment. The
resulting interval is narrow and wrong. This is bias, not variance, so a larger
sample does not rescue it; the estimate converges to the wrong number.

**Fit `f` with a flexible method and plug it in.** Now the regularisation that
made the flexible fit work bleeds into `b`. Shrinking `f` toward zero leaves
confounding in the residual and biases `b` away from the truth; loosening it
lets `f` absorb variation that belongs to `x'b` and attenuates `b` toward zero.
There is no setting of the penalty that threads between them, because the two
failures sit on opposite sides of the answer. This is
regularisation-induced confounding, and it is why plugging a good predictor
into an inferential slot does not work.

`bllnn` addresses the second problem the way the double machine learning
literature does -- cross-fitting and Robinson partialling-out -- and gets
uncertainty in `b` from an exact conditional draw rather than an asymptotic
approximation.

## What it actually does

Three ingredients, each doing one job.

**A frozen feature map.** A network is trained on held-out folds and then
fixed. Write the learned last hidden layer as `Phi`, an `n x m` matrix. Because
`Phi` does not depend on the response being fitted, the last-layer coefficients
have an exactly Gaussian conditional posterior. This is sample splitting, not
an optimisation: it is what makes the next step exact rather than approximate.

**An exact conjugate draw.** With prior `w ~ N(0, tau2 I)` and residual `r`,

```
w | r, sigma  ~  N( V Phi'r / sigma^2 ,  V ),     V = (Phi'Phi / sigma^2 + I / tau2)^-1
```

drawn in closed form. No inner MCMC, no variational step, no optimiser. The
prior is proper, so this is well defined even when `m > n`.

**Partialling-out.** The network is given `E[x | z]` as an extra feature so it
can represent the confounding channel, and `x` is then residualised against it.
Without the residualisation, `Phi` contains `E[x | z]` while `y` depends on it
only through `x'b`; the two compete for the same direction and `b` is
attenuated toward `b(1 - rho^2)`.

Around these, a Gibbs sampler alternates `b | f, sigma`, `f | b, sigma` and
`sigma^2 | b, f`. The noise scale is always owned by the caller and never
estimated inside the block, which would use the data twice.

## Installation

```r
# install.packages("remotes")
remotes::install_github("sapniyogi/uncertainty-quantifier-bllnn")
```

Requires a C++ toolchain (Rtools on Windows). CRAN release pending.

## Quick start

```r
library(bllnn)

sim <- sim_partial_linear(n = 400, beta = c(treat = 1.5), p_z = 5,
                          confounding = 0.6, seed = 1)

fit <- bllnn(y ~ z1 + z2 + z3 + z4 + z5,   # what f(z) is over
             data = sim$data,
             linear = ~ treat,             # what you want a coefficient for
             seed = 1)

summary(fit)
```

```
Linear coefficients (95% credible intervals):

      Estimate     SD  Lower  Upper ESS
treat   1.4736 0.0938 1.2939 1.6620 826 *

* interval excludes zero

residual sd    : 1.3672
sd of f(Z)     : 5.0064
observations   : 400
cross-fitting  : 5 folds, 52 features per fold
draws          : 1500 kept of 2000
prior variance : 5.132 (auto)
smallest ESS   : 826 of 1500 draws
```

The truth is 1.5 and the interval covers it. Ordinary least squares on the same
data returns **1.865**: the curvature in `f(z)` has landed on the coefficient.

The formula split is the only thing you must get right. The main formula names
the variables `f` is over; `linear` names the terms you want inference for.
Nothing is guessed.

Standard methods behave as expected:

```r
coef(fit)                                  # posterior means
confint(fit, level = 0.9)                  # credible intervals
predict(fit, type = "f")                   # the fitted nonlinear part
predict(fit, newdata = new_df)             # response on new rows
plot(fit)                                  # trace and posterior density
```

## Using it inside your own sampler

This is the part that distinguishes the package. `bllnn()` is a convenience
wrapper over an object you can drive yourself:

```r
cf   <- bllnn_crossfit(z, y, linear = x, folds = 5)   # learn features once
mod  <- bllnn_sampler(cf, tau2 = "auto")
xr   <- partial_out(cf)                               # x - E[x | z]

stopifnot(is_valid_kernel(mod))

for (s in 1:n_iter) {
  # --- your draw of b given f and sigma ---

  set_response(mod, y - as.vector(xr %*% b))  # the residual this block explains
  set_sigma(mod, sigma)                       # you own the noise scale
  f <- gibbs_step(mod)                        # one exact posterior draw of f

  # --- your draw of sigma given b and f ---
}
```

The object has reference semantics, so nothing is reassigned each sweep. The
contract is deliberately shaped like `dbarts` (`setResponse` / `setSigma` /
`run`), so if you have written a BART Gibbs sampler you already know this one.

Three properties make that loop legitimate rather than merely convenient:

1. **The draw is exact.** Conditional on the frozen features it is the true
   Gaussian conditional, in closed form.
2. **You own `sigma`.** Estimating the noise scale inside the block would use
   the data twice and break the joint chain.
3. **Invalid kernels are refused.** Not every posterior/feature combination is
   a legitimate Gibbs transition. `gibbs_step()` errors unless
   `is_valid_kernel(mod)` is `TRUE`, overridable only with an explicit
   `force = TRUE`. Producing plausible output that targets the wrong
   distribution is the worst failure available here, so it is made loud rather
   than left to the user to notice.

`vignette("custom-sampler", package = "bllnn")` works a complete host sampler
end to end, including the binary case.

### Bringing your own features

`bllnn_sampler()` accepts any numeric matrix, so the network is not
load-bearing for the contract. If you already have a feature map -- a `torch`
model, a spline basis, random projections, a pre-trained body -- compute it and
hand it over:

```r
Phi <- my_feature_map(z)            # n x m, however you produced it
mod <- bllnn_sampler(Phi, tau2 = "auto")
```

Everything downstream is identical. What is **not** currently swappable is the
cross-fitting path: `bllnn_crossfit()` calls the built-in network directly and
takes no `fitter` argument, so cross-fitted features from your own model mean
writing the fold loop, and the `E[x | z]` auxiliaries, yourself. That is a known
gap rather than a decision; say so on the issue tracker if it blocks you.

The built-in network is configurable: `width` takes a vector for arbitrary
depth (`width = c(32, 16, 8)`), plus `activation`, `learn_rate`,
`weight_decay`, `validation`, `patience` and `tune`. These pass through
`bllnn_crossfit()` and `bllnn()` unchanged.

## Non-Gaussian outcomes

| `posterior =` | Valid Gibbs kernel | Use for |
|---|---|---|
| `"conjugate"` | yes, exact | Gaussian outcomes. The default |
| `"polyagamma"` | yes, exact | Binary outcomes, and negative-binomial counts via `dispersion` |
| `"laplace"` | **no** -- approximate | Poisson counts, which the Polya-Gamma identity cannot reach, and post-hoc uncertainty on a body trained outside any sampler |

Binary and negative-binomial outcomes use Polya-Gamma augmentation (Polson,
Scott and Windle, 2013), which is exact at both steps: conditional on the
latent variables the likelihood is Gaussian in the weights, so the block stays
a valid kernel. `rpolyagamma()` implements Devroye's alternating-series
sampler, so nothing is truncated.

Under a non-identity link there is no scale on which you can subtract your own
contribution, so the host passes it as an offset via `set_offset()` and the
block conditions on it. `gibbs_step()` leaves the latent variables on the
object as `mod$omega`, which your coefficient draw needs.

`"laplace"` is approximate and **is not a kernel** -- `gibbs_step()` refuses it
without `force = TRUE`. For a Gaussian outcome it coincides with the exact
conjugate draw to machine precision; elsewhere its error is measured rather
than assumed, against a long Polya-Gamma run
(`inst/validation/laplace_checks.R`).

## What you get, and what you don't

**Validated.** Over 100 simulated datasets at confounding 0.6, nominal 95%
credible intervals covered the true coefficient **96%** of the time, with bias
**-0.024** where naive least squares carried **+0.397**. Effective sample size
was 575 of 800 kept draws. The script is
`inst/validation/coverage_simulation.R`, and it exercises the configuration the
defaults give you rather than a tuned variant.

**Stated plainly, because they are real:**

- **Pointwise credible bands for `f(z)` undercover.** They quantify uncertainty
  about the best fit *within the span of the frozen features*, and say nothing
  about the distance from that span to the truth. On a smooth test problem,
  nominal 95% bands covered `f` at 58%. This is a property of the
  frozen-feature design, not of any posterior -- the exact conjugate path gives
  57.7% and the Laplace path 58.3% on the same problem. **The coverage claim
  above is about `b`, which is a different quantity resting on a different
  argument.** If your target is `f` itself rather than `b`, this is the wrong
  tool.
- **A Gaussian last layer gives a Gaussian predictive distribution.** It cannot
  represent skewed or multimodal conditionals.
- **Not a predictive-accuracy story.** Bayesian last layer methods generally do
  not beat Gaussian processes on standard regression benchmarks, and this
  package is not pitched on that. The claim is composability and valid
  conditional draws.
- **A hand-set `prior_beta` must be on the scale of your outcome.** The default
  is `"auto"` and scales itself; a number you pass does not. The fixed default
  that `"auto"` replaced returned an effect of \$1 with an interval of
  [-\$19, \$22] on earnings data, because the prior outweighed the likelihood
  by three orders of magnitude.
- **Negative-binomial dispersion is fixed, not estimated.** It must be a
  positive integer; profile over a few values.
- **The rate at which residual bias vanishes with `n` is not claimed.**
  Residualising was better at every sample size tested, but the experiments run
  so far cannot establish the rate, so no rate is asserted.

## How this relates to other packages

Honestly, and with the cases where you should use something else.

- **`dbarts` / BART.** The API model for this package, and the closest
  relative in spirit. BART is a better-established and more thoroughly
  benchmarked function class. Use BART if you want the tree prior and its
  track record; use this if you specifically want a neural basis or need the
  block inside a larger Bayesian model.
- **`DoubleML`, `grf`.** Frequentist semiparametric inference with
  cross-fitting, mature and broad in their choice of learners. If you want a
  point estimate with a standard error from a well-trodden framework, they are
  the safer default. `bllnn` differs by producing a full posterior that
  composes with other Bayesian model components.
- **`mgcv` / GAMs.** If an additive structure is plausible for `f(z)`, a GAM
  is more interpretable, faster, and has far better diagnostics. Reach for a
  network only when you have reason to doubt additivity.
- **`bartCause`, `stochtree`.** Causal BART workflows. Similar goals,
  tree-based, more complete as end-to-end causal tooling.

The niche `bllnn` is actually aimed at is the one where you are writing your
own sampler and need *one* conditional draw of a nonlinear function that you
can defend. If you are not writing your own sampler, the convenience wrapper
works, but you should weigh the alternatives above seriously.

## Checking your fit

This is MCMC, so the output is a chain and deserves inspection.

```r
summary(fit)    # reports effective sample size per coefficient
plot(fit)       # trace must look like noise; density should be unimodal
```

`summary()` warns when a chain has not mixed. Effective sample size, not the
number of iterations, is what governs interval accuracy; a few hundred is
enough for a 95% interval, and if it is much lower, raise `n_iter` before
believing the interval.

## Learn more

```r
vignette("bllnn",          package = "bllnn")   # the causal use case
vignette("custom-sampler", package = "bllnn")   # writing your own host sampler
vignette("lalonde",        package = "bllnn")   # a real dataset, honestly
```

The LaLonde vignette is the one to read if you are sceptical: it runs on data
with poor overlap and a non-Gaussian outcome, states its violated assumptions,
and reports an interval that includes zero where the linear model's does not.

## References

- Polson, N. G., Scott, J. G. and Windle, J. (2013). Bayesian inference for
  logistic models using Polya-Gamma latent variables. *JASA* 108(504),
  1339-1349.
- Hahn, P. R., Murray, J. S. and Carvalho, C. M. (2020). Bayesian regression
  tree models for causal inference. *Bayesian Analysis* 15(3), 965-1056.
- Chernozhukov, V. et al. (2018). Double/debiased machine learning for
  treatment and structural parameters. *The Econometrics Journal* 21(1),
  C1-C68.
- Robinson, P. M. (1988). Root-N-consistent semiparametric regression.
  *Econometrica* 56(4), 931-954.

## Contributing

Issues and pull requests are welcome at the
[tracker](https://github.com/sapniyogi/uncertainty-quantifier-bllnn/issues).
If you are reporting a statistical problem rather than a bug, a reproducible
simulation is worth more than a description -- most of the real defects in this
package were found that way.

## License

GPL-3.
