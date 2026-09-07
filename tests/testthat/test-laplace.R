# The Laplace path is the one approximate method in the package, so the tests
# have two jobs rather than one: check that it is right where it claims to be
# exact, and measure how wrong it is where it does not.
#
# Every yardstick here is an independent implementation. The Gaussian case is
# checked against conjugate_moments(), which reaches the same answer by a
# closed form rather than by iterating. The binomial and Poisson cases are
# checked against glm(), whose IRLS and whose vcov() share no code with this
# one. Checking the Newton iteration against itself would prove nothing.

# --- shared helpers ---------------------------------------------------------

make_design <- function(n, m, seed = 1) {
  set.seed(seed)
  matrix(stats::rnorm(n * m), n, m)
}

# glm() stops when the relative deviance change falls below 1e-8, which for a
# binomial fit leaves the coefficients right to about 4e-8 but the covariance
# only to 4e-3 -- the weight mu(1 - mu) is far more sensitive to a small
# coefficient error than the fit is. Comparing against a yardstick that has
# stopped short measures glm's stopping rule rather than this implementation,
# so the tolerance is tightened until glm has genuinely converged. It then
# needs one more iteration and agrees to 1e-7.
tight <- stats::glm.control(epsilon = 1e-12, maxit = 200)

# --- exactness on the Gaussian case -----------------------------------------

test_that("for a Gaussian outcome the Laplace moments are the conjugate ones", {
  # This is the correctness test CLAUDE.md asks for against laplace-torch,
  # done internally: for a Gaussian likelihood the log posterior is exactly
  # quadratic, so fitting a quadratic at the mode is not an approximation and
  # the two paths must agree to machine precision, not merely closely.
  Phi <- make_design(300, 5, seed = 11)
  r <- as.vector(Phi %*% c(1, -2, 0.5, 0, 0.3)) + stats::rnorm(300, sd = 0.7)

  lap <- laplace_moments(Phi, r, tau2 = 4, family = "gaussian", sigma = 0.7)
  cnj <- conjugate_moments(Phi, r, sigma = 0.7, tau2 = 4)

  expect_equal(unname(lap$mean), unname(cnj$mean), tolerance = 1e-12)
  expect_equal(unname(lap$cov), unname(cnj$cov), tolerance = 1e-12)
  expect_equal(unname(lap$precision), unname(cnj$precision), tolerance = 1e-12)
})

test_that("the Gaussian case converges in one step, as a quadratic must", {
  # Two iterations reported: one that moves to the mode, one that finds it has
  # nowhere left to go. If this ever climbs, the log posterior being optimised
  # is not the quadratic it is supposed to be.
  Phi <- make_design(200, 4, seed = 12)
  r <- as.vector(Phi %*% c(1, 0, -1, 2)) + stats::rnorm(200, sd = 0.5)

  expect_equal(
    laplace_moments(Phi, r, tau2 = 2, family = "gaussian",
                    sigma = 0.5)$iterations,
    2L)
})

test_that("the Gaussian mode does not depend on where the iteration starts", {
  # The starting point is hard-coded at zero, so this exercises the same claim
  # from the other side: shifting the problem by a known offset must shift the
  # mode by exactly the corresponding amount and leave the covariance alone.
  Phi <- make_design(200, 3, seed = 13)
  r <- as.vector(Phi %*% c(1, -1, 0.5)) + stats::rnorm(200, sd = 0.4)
  o <- stats::rnorm(200)

  with_offset <- laplace_moments(Phi, r, tau2 = 3, family = "gaussian",
                                 sigma = 0.4, offset = o)
  shifted <- laplace_moments(Phi, r - o, tau2 = 3, family = "gaussian",
                             sigma = 0.4)

  expect_equal(unname(with_offset$mean), unname(shifted$mean),
               tolerance = 1e-12)
  expect_equal(unname(with_offset$cov), unname(shifted$cov), tolerance = 1e-12)
})

# --- the non-Gaussian families, against glm ---------------------------------

test_that("the binomial MAP is the glm fit as the prior flattens", {
  # With tau2 enormous the ridge is negligible and the MAP is the MLE, so a
  # completely independent implementation has to land in the same place. The
  # covariance is the inverse observed information, which is what vcov() on a
  # canonical-link glm returns.
  Phi <- make_design(300, 5, seed = 21)
  y <- stats::rbinom(300, 1, stats::plogis(
    as.vector(Phi %*% c(0.8, -0.5, 0.3, 0, 0.2))))

  lap <- laplace_moments(Phi, y, tau2 = 1e10, family = "binomial")
  fit <- stats::glm(y ~ Phi - 1, family = stats::binomial(),
                    control = tight)

  expect_equal(unname(lap$mean), unname(stats::coef(fit)), tolerance = 1e-7)
  expect_equal(unname(lap$cov), unname(stats::vcov(fit)), tolerance = 1e-6)
})

test_that("the Poisson MAP is the glm fit as the prior flattens", {
  # Poisson is the reason this path exists: the Polya-Gamma identity needs the
  # (1 + e^psi)^-b form, which the Poisson likelihood does not have, so the
  # exact augmentation cannot reach it at all.
  Phi <- make_design(300, 5, seed = 22)
  y <- stats::rpois(300, exp(as.vector(Phi %*% c(0.4, -0.3, 0.2, 0, 0.1))))

  lap <- laplace_moments(Phi, y, tau2 = 1e10, family = "poisson")
  fit <- stats::glm(y ~ Phi - 1, family = stats::poisson(),
                    control = tight)

  expect_equal(unname(lap$mean), unname(stats::coef(fit)), tolerance = 1e-7)
  expect_equal(unname(lap$cov), unname(stats::vcov(fit)), tolerance = 1e-6)
})

test_that("an offset is handled the way glm handles one", {
  Phi <- make_design(250, 4, seed = 23)
  o <- stats::rnorm(250, sd = 0.5)
  y <- stats::rpois(250, exp(o + as.vector(Phi %*% c(0.3, -0.2, 0.1, 0))))

  lap <- laplace_moments(Phi, y, tau2 = 1e10, family = "poisson", offset = o)
  fit <- stats::glm(y ~ Phi - 1, family = stats::poisson(), offset = o,
                    control = tight)

  expect_equal(unname(lap$mean), unname(stats::coef(fit)), tolerance = 1e-7)
})

test_that("the prior shrinks, and shrinks more as tau2 falls", {
  Phi <- make_design(150, 4, seed = 24)
  y <- stats::rbinom(150, 1, stats::plogis(
    as.vector(Phi %*% c(1.5, -1, 0.5, 0))))

  norms <- vapply(c(0.01, 0.1, 1, 100), function(t2) {
    sqrt(sum(laplace_moments(Phi, y, tau2 = t2, family = "binomial")$mean^2))
  }, numeric(1))

  expect_true(all(diff(norms) > 0))
})

test_that("separation still gives a finite answer, because the prior is proper", {
  # Separation sends the MLE to infinity. With a proper prior the MAP is still
  # finite, so this must succeed -- it is the prior, not the data, that makes
  # it well posed, and that is worth pinning down.
  Phi <- cbind(seq(-2, 2, length.out = 40))
  y <- as.numeric(Phi[, 1] > 0)

  fit <- laplace_moments(Phi, y, tau2 = 1, family = "binomial")
  expect_true(all(is.finite(fit$mean)))
  expect_true(all(is.finite(fit$cov)))
})

# --- the draw ---------------------------------------------------------------

test_that("draws from the fitted moments have those moments", {
  # Split deliberately from the test below. laplace_draw() is
  # draw_from_moments(laplace_moments(...)), and those two halves fail in
  # different ways, so they are checked separately rather than through 20000
  # calls to the composition -- which would also re-run the whole Newton
  # optimisation of one fixed problem 20000 times, for nothing.
  Phi <- make_design(200, 3, seed = 31)
  y <- stats::rbinom(200, 1, stats::plogis(as.vector(Phi %*% c(1, -0.5, 0.25))))
  post <- laplace_moments(Phi, y, tau2 = 5, family = "binomial")

  set.seed(31)
  n_draws <- 20000
  W <- t(replicate(n_draws, draw_from_moments(post)))

  # Bounds from sampling theory, as elsewhere in this suite: the draws are iid
  # Gaussian with known moments, so five standard errors is a real bound and
  # not a number chosen to pass.
  se_mean <- sqrt(diag(post$cov) / n_draws)
  expect_true(all(abs(colMeans(W) - post$mean) < 5 * se_mean))

  V <- stats::cov(W)
  se_cov <- outer(seq_len(3), seq_len(3), Vectorize(function(i, j) {
    sqrt((post$cov[i, i] * post$cov[j, j] + post$cov[i, j]^2) / n_draws)
  }))
  expect_true(all(abs(V - post$cov) < 5 * se_cov))
})

test_that("laplace_draw is exactly that draw, on those moments", {
  # The other half: that laplace_draw() hands the right moments to the right
  # sampler. Checked by reproducing it exactly from the same seed rather than
  # statistically, which is both stronger and instant.
  Phi <- make_design(150, 4, seed = 33)
  y <- stats::rpois(150, exp(as.vector(Phi %*% c(0.3, -0.2, 0.1, 0))))
  post <- laplace_moments(Phi, y, tau2 = 5, family = "poisson")

  set.seed(99)
  from_draw <- laplace_draw(Phi, y, tau2 = 5, family = "poisson")
  set.seed(99)
  from_moments <- draw_from_moments(post, colnames(Phi))

  expect_identical(from_draw, from_moments)
})

test_that("consecutive draws differ, and carry the column names", {
  Phi <- make_design(100, 3, seed = 32)
  colnames(Phi) <- c("a", "b", "c")
  y <- stats::rpois(100, exp(as.vector(Phi %*% c(0.2, -0.1, 0.05))))

  set.seed(1)
  d1 <- laplace_draw(Phi, y, tau2 = 5, family = "poisson")
  d2 <- laplace_draw(Phi, y, tau2 = 5, family = "poisson")

  expect_false(identical(d1, d2))
  expect_identical(names(d1), c("a", "b", "c"))
})

# --- the guard --------------------------------------------------------------

test_that("laplace is not a valid kernel and gibbs_step says so", {
  # Design rule 4. The approximation is usable one-shot and must never be
  # iterated, so the refusal is the feature.
  Phi <- make_design(120, 3, seed = 41)
  y <- stats::rbinom(120, 1, 0.5)

  mod <- bllnn_sampler(Phi, tau2 = 1, posterior = "laplace",
                       family = "binomial")
  expect_false(is_valid_kernel(mod))
  expect_match(attr(is_valid_kernel(mod), "reason"), "not the")

  set_response(mod, y)
  expect_error(gibbs_step(mod), "not a valid Gibbs")

  f <- gibbs_step(mod, force = TRUE)
  expect_length(f, 120)
  expect_true(all(is.finite(f)))
})

test_that("the sampler's laplace path agrees with the function it wraps", {
  Phi <- make_design(150, 4, seed = 42)
  y <- stats::rbinom(150, 1, stats::plogis(
    as.vector(Phi %*% c(0.5, -0.5, 0.25, 0))))

  mod <- bllnn_sampler(Phi, tau2 = 2, posterior = "laplace",
                       family = "binomial")
  set_response(mod, y)

  set.seed(7)
  from_sampler <- gibbs_step(mod, force = TRUE)
  direct <- laplace_moments(Phi, y, tau2 = 2, family = "binomial")

  # The draw is random, but the mode behind it is not: the weights the sampler
  # stored must come from the same Gaussian, so the sampler's centre is the
  # function's centre.
  expect_equal(unname(mod$w), unname(direct$mean),
               tolerance = 4 * max(sqrt(diag(direct$cov))))
  expect_length(from_sampler, 150)
})

test_that("the Gaussian family still needs the host's sigma", {
  # Design rule 2 does not lapse because the posterior changed.
  Phi <- make_design(100, 3, seed = 43)
  mod <- bllnn_sampler(Phi, tau2 = 1, posterior = "laplace")
  set_response(mod, stats::rnorm(100))

  expect_error(gibbs_step(mod, force = TRUE), "No noise level set")

  set_sigma(mod, 1)
  expect_length(gibbs_step(mod, force = TRUE), 100)
})

test_that("the non-Gaussian families do not ask for a sigma they cannot use", {
  Phi <- make_design(100, 3, seed = 44)
  mod <- bllnn_sampler(Phi, tau2 = 1, posterior = "laplace",
                       family = "poisson")
  set_response(mod, stats::rpois(100, 2))
  expect_length(gibbs_step(mod, force = TRUE), 100)
})

# --- what the arguments refuse ----------------------------------------------

test_that("sigma is required exactly where it means something", {
  Phi <- make_design(50, 2, seed = 51)
  r <- stats::rnorm(50)

  expect_error(laplace_moments(Phi, r, tau2 = 1, family = "gaussian"),
               "`sigma` is required")
  expect_error(
    laplace_moments(Phi, stats::rbinom(50, 1, 0.5), tau2 = 1,
                    family = "binomial", sigma = 1),
    "applies only to family")
  expect_error(
    laplace_moments(Phi, r, tau2 = 1, family = "gaussian", sigma = -1),
    "standard deviation")
})

test_that("the outcome has to match the family", {
  Phi <- make_design(50, 2, seed = 52)

  expect_error(laplace_moments(Phi, stats::rnorm(50), tau2 = 1,
                               family = "binomial"),
               "must be 0 or 1")
  expect_error(laplace_moments(Phi, stats::rnorm(50), tau2 = 1,
                               family = "poisson"),
               "non-negative integers")
  expect_error(laplace_moments(Phi, rep(0, 50), tau2 = 1, family = "gamma"),
               "`family` must be one of")
})

test_that("the sampler catches a residual passed where an outcome belongs", {
  # The same trap the Polya-Gamma path guards: under a non-identity link there
  # is no scale on which the host can subtract its own contribution first.
  Phi <- make_design(60, 3, seed = 53)

  mod <- bllnn_sampler(Phi, tau2 = 1, posterior = "laplace",
                       family = "binomial")
  expect_error(set_response(mod, stats::rnorm(60)), "must be 0/1")

  mod2 <- bllnn_sampler(Phi, tau2 = 1, posterior = "laplace",
                        family = "poisson")
  expect_error(set_response(mod2, stats::rnorm(60)), "non-negative counts")
})

test_that("family belongs to the laplace posterior and nowhere else", {
  Phi <- make_design(50, 2, seed = 54)

  expect_error(bllnn_sampler(Phi, tau2 = 1, family = "binomial"),
               "applies only to posterior")
  expect_error(
    bllnn_sampler(Phi, tau2 = 1, posterior = "laplace", family = "gamma"),
    "`family` must be one of")
  expect_null(bllnn_sampler(Phi, tau2 = 1)$family)
  expect_identical(
    bllnn_sampler(Phi, tau2 = 1, posterior = "laplace")$family, "gaussian")
})

test_that("other arguments are validated", {
  Phi <- make_design(50, 2, seed = 55)
  y <- stats::rbinom(50, 1, 0.5)

  expect_error(laplace_moments(Phi, y[1:10], tau2 = 1, family = "binomial"),
               "has length 10")
  expect_error(laplace_moments(Phi, y, tau2 = -1, family = "binomial"),
               "prior variance")
  expect_error(laplace_moments(Phi, y, tau2 = 1, family = "binomial",
                               offset = rep(0, 3)),
               "of length 1 or 50")
})

test_that("the print method names the family", {
  Phi <- make_design(50, 2, seed = 56)
  mod <- bllnn_sampler(Phi, tau2 = 1, posterior = "laplace",
                       family = "poisson")
  expect_output(print(mod), "laplace \\(poisson\\)")
  expect_output(print(bllnn_sampler(Phi, tau2 = 1)), "posterior  : conjugate")
})

# --- how much the approximation costs ---------------------------------------

test_that("the Laplace posterior is close to the exact one, and optimistic", {
  # The only test here with a ground truth rather than a reference: the
  # Polya-Gamma sampler targets the exact logistic posterior, so a long run of
  # it says what the Laplace fit is approximating. This is what justifies
  # preferring posterior = "polyagamma" for binary outcomes, and it is asserted
  # rather than described so that a regression in either direction is caught.
  #
  # Two claims, both of which follow from the posterior being log-concave with
  # tails heavier than Gaussian. The mode is near but not at the mean, and a
  # Gaussian matched to the curvature at the mode is too narrow. Neither is a
  # defect to be fixed; they are what a Laplace approximation is.
  skip_on_cran()

  set.seed(9)
  n <- 80
  m <- 6
  Phi <- matrix(stats::rnorm(n * m), n, m)
  y <- stats::rbinom(n, 1, stats::plogis(
    as.vector(Phi %*% stats::rnorm(m))))

  lap <- laplace_moments(Phi, y, tau2 = 4, family = "binomial")

  mod <- bllnn_sampler(Phi, tau2 = 4, posterior = "polyagamma")
  set_response(mod, y)
  n_draws <- 4000
  W <- matrix(0, n_draws, m)
  for (i in seq_len(n_draws)) {
    gibbs_step(mod)
    W[i, ] <- mod$w
  }
  W <- W[-seq_len(n_draws / 4), , drop = FALSE]

  exact_mean <- colMeans(W)
  exact_sd <- apply(W, 2, stats::sd)
  gap <- abs(lap$mean - exact_mean) / exact_sd
  ratio <- sqrt(diag(lap$cov)) / exact_sd

  # Close enough to be useful. Measured at 0.09 to 0.36 posterior standard
  # deviations here, and shrinking with n: 0.19 to 0.21 at n = 300.
  expect_true(max(gap) < 0.75)

  # But not exact, which is the whole reason the augmentation exists. If this
  # ever fails, either the yardstick has stopped being the exact posterior or
  # the two have been wired to the same code.
  expect_true(max(gap) > 0.02)

  # Optimistic, in the direction theory predicts. Measured at 0.91 to 0.98,
  # against a Monte Carlo error on exact_sd of under 2% at this chain length.
  # Reported credible intervals are correspondingly too narrow.
  expect_true(all(ratio < 1))
  expect_true(all(ratio > 0.8))
})

# --- robustness of the iteration --------------------------------------------

test_that("the iteration converges across many random problems", {
  # A fuzz test rather than a fixed case: the iteration has a step-halving
  # loop and a hard failure at its floor, and neither is exercised by a
  # hand-picked dataset that happens to converge in four steps.
  #
  # Each problem is checked against the definition of the answer rather than
  # against the loop's own opinion of it -- the gradient of the log posterior
  # must vanish at the reported mode. A silent early exit, a stall, or a step
  # rule that stops short all fail that check.
  #
  # The floating-point hazard the slack in the acceptance test guards was
  # measured rather than assumed: over these same 100 problems the strict rule
  # it replaced never hit the halving floor. The slack is insurance against a
  # mechanism that is real but rare, not a fix for an observed failure.
  skip_on_cran()

  set.seed(2026)
  for (i in seq_len(100)) {
    n <- sample(40:200, 1)
    m <- sample(2:8, 1)
    Phi <- matrix(stats::rnorm(n * m), n, m)
    w0 <- stats::rnorm(m, sd = 0.8)
    eta <- as.vector(Phi %*% w0)
    tau2 <- sample(c(0.5, 2, 10, 100), 1)

    if (i %% 2 == 0) {
      y <- stats::rbinom(n, 1, stats::plogis(eta))
      fam <- "binomial"
    } else {
      y <- stats::rpois(n, exp(pmin(eta, 3)))
      fam <- "poisson"
    }

    fit <- expect_no_error(
      expect_no_warning(laplace_moments(Phi, y, tau2 = tau2, family = fam)))
    expect_true(all(is.finite(fit$mean)))
    expect_true(all(is.finite(fit$cov)))

    # Converged means the gradient of the log posterior vanishes there. This
    # checks the answer rather than the loop's own opinion of the answer.
    mu <- if (fam == "binomial") stats::plogis(as.vector(Phi %*% fit$mean)) else
      exp(as.vector(Phi %*% fit$mean))
    grad <- as.vector(crossprod(Phi, y - mu) - fit$mean / tau2)
    expect_lt(max(abs(grad)), 1e-6)
  }
})
