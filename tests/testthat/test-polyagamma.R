# --- the sampler against closed forms ---------------------------------------

test_that("draws match the analytic mean and variance across z", {
  # There is no reference implementation in the package to test against, so
  # the closed-form moments are the check. Deviations are reported in Monte
  # Carlo standard errors, which is the only scale on which "close" means
  # anything.
  set.seed(1)
  n <- 20000

  for (z in c(0, 0.5, 1, 2, 5, 10)) {
    d <- rpolyagamma(n, z)
    se_mean <- sqrt(pg_var(z) / n)

    expect_lt(abs(mean(d) - pg_mean(z)), 4 * se_mean,
              label = paste("mean at z =", z))
    # The variance of a sample variance is dominated by the fourth moment,
    # which is not in closed form here, so this is a proportional bound.
    expect_equal(var(d), pg_var(z), tolerance = 0.06,
                 info = paste("variance at z =", z))
  }
})

test_that("the analytic moments have the right limits at zero", {
  expect_equal(pg_mean(0), 0.25)
  expect_equal(pg_var(0), 1 / 24)
  # Continuous approaching zero rather than only defined there.
  expect_equal(pg_mean(1e-6), 0.25, tolerance = 1e-8)
  expect_equal(pg_var(1e-6), 1 / 24, tolerance = 1e-6)
  # Symmetric in z.
  expect_equal(pg_mean(-3), pg_mean(3))
  expect_equal(pg_var(-3), pg_var(3))
})

test_that("the sampler agrees distributionally with the series construction", {
  # An independent construction of the same law:
  #   PG(1, z) = (1 / (2 pi^2)) sum_k g_k / ((k - 1/2)^2 + z^2 / (4 pi^2))
  # with g_k iid Exp(1). Truncating it is exactly the approximation the exact
  # sampler exists to avoid, but with 4000 terms it is accurate enough to be a
  # yardstick, and it shares no code with Devroye's method. Agreement across
  # the whole quantile range is far stronger evidence than matching moments.
  series_draw <- function(z, n, terms = 4000) {
    k <- seq_len(terms)
    d <- (k - 0.5)^2 + z^2 / (4 * pi^2)
    vapply(seq_len(n), function(i) {
      sum(stats::rexp(terms) / d) / (2 * pi^2)
    }, numeric(1))
  }

  set.seed(2)
  for (z in c(0, 2)) {
    a <- rpolyagamma(4000, z)
    b <- series_draw(z, 4000)
    probs <- c(0.1, 0.25, 0.5, 0.75, 0.9)
    qa <- stats::quantile(a, probs)
    qb <- stats::quantile(b, probs)
    expect_equal(unname(qa), unname(qb), tolerance = 0.05,
                 info = paste("quantiles at z =", z))
  }
})

test_that("draws are positive and finite", {
  set.seed(3)
  for (z in c(0, 1, 8, 30)) {
    d <- rpolyagamma(500, z)
    expect_true(all(d > 0), label = paste("positive at z =", z))
    expect_true(all(is.finite(d)), label = paste("finite at z =", z))
  }
})

test_that("large tilts do not stall or overflow", {
  # z far out is where the two-piece proposal and the exponential terms are
  # most likely to misbehave.
  set.seed(4)
  d <- rpolyagamma(200, 100)
  expect_true(all(is.finite(d)))
  expect_lt(abs(mean(d) - pg_mean(100)), 0.01)
})

test_that("z is recycled and lengths are honoured", {
  set.seed(5)
  expect_length(rpolyagamma(10, 0), 10L)
  expect_length(rpolyagamma(z = c(0, 1, 2)), 3L)
  expect_length(rpolyagamma(6, c(0, 5)), 6L)
  expect_length(rpolyagamma(0), 0L)

  # A vector z really does tilt each draw, rather than using the first value.
  set.seed(6)
  big <- rpolyagamma(4000, rep(c(0, 10), each = 2000))
  expect_lt(abs(mean(big[1:2000]) - pg_mean(0)), 0.01)
  expect_lt(abs(mean(big[2001:4000]) - pg_mean(10)), 0.01)
})

test_that("a fixed seed reproduces the draws", {
  set.seed(7); a <- rpolyagamma(50, 1.5)
  set.seed(7); b <- rpolyagamma(50, 1.5)
  set.seed(8); d <- rpolyagamma(50, 1.5)

  expect_identical(a, b)
  expect_false(isTRUE(all.equal(a, d)))
})

test_that("arguments are validated", {
  expect_error(rpolyagamma(-1), "non-negative integer")
  expect_error(rpolyagamma(2.5), "non-negative integer")
  expect_error(rpolyagamma(c(1, 2)), "non-negative integer")
  expect_error(rpolyagamma(5, z = "a"), "numeric")
  expect_error(rpolyagamma(5, z = c(1, NA)), "no NAs")
  expect_warning(rpolyagamma(5, c(1, 2)), "recycled")
})

# --- the pieces the sampler is built from -----------------------------------

test_that("the truncated inverse Gaussian respects its bound", {
  # Both regimes: mean above the truncation point, where the draw comes from a
  # truncated-normal representation, and mean below it, where plain rejection
  # is used. A draw above the bound from either would silently bias the
  # acceptance step that follows.
  set.seed(9)
  for (z in c(0, 0.5, 5, 20)) {
    x <- vapply(1:400, function(i) pg_rtigauss(z), numeric(1))
    expect_true(all(x > 0 & x <= PG_TRUNC),
                label = paste("bounded at z =", z))
  }
})

test_that("both series branches give the same density at the split point", {
  # pg_coef switches formula at t. The two expressions are NOT the same
  # function term by term -- they are different series, related by a Jacobi
  # theta transformation, and Devroye uses whichever converges faster on each
  # side. What must agree is the alternating sum, which is the density. A
  # mismatch there would put a real discontinuity inside the acceptance test.
  t <- PG_TRUNC
  n <- 0:40
  sign <- (-1)^n

  small <- sum(sign * pi * (n + 0.5) * (2 / (pi * t))^1.5 *
                 exp(-2 * (n + 0.5)^2 / t))
  large <- sum(sign * pi * (n + 0.5) * exp(-(n + 0.5)^2 * pi^2 * t / 2))

  expect_equal(small, large, tolerance = 1e-8)

  # The plain sums do not agree, which is why the alternating sign matters.
  plain_small <- sum(pi * (n + 0.5) * (2 / (pi * t))^1.5 *
                       exp(-2 * (n + 0.5)^2 / t))
  plain_large <- sum(pi * (n + 0.5) * exp(-(n + 0.5)^2 * pi^2 * t / 2))
  expect_gt(abs(plain_small - plain_large), 1e-5)
})

test_that("the inverse Gaussian distribution function handles the zero limit", {
  # mu = Inf is reached whenever z is zero, which happens on the very first
  # sweep when the weights start at zero.
  expect_equal(pg_pigauss(PG_TRUNC, Inf),
               2 * pnorm(-1 / sqrt(PG_TRUNC)))
  expect_true(is.finite(pg_pigauss(PG_TRUNC, 1e6)))
  expect_equal(pg_pigauss(PG_TRUNC, 1e12), pg_pigauss(PG_TRUNC, Inf),
               tolerance = 1e-6)
})

# --- the posterior path -----------------------------------------------------

pg_case <- function(n = 400, seed = 1) {
  set.seed(seed)
  Phi <- cbind(intercept = 1, matrix(rnorm(n * 3), n, 3,
                                     dimnames = list(NULL, paste0("h", 1:3))))
  w <- c(-0.3, 1.2, -0.8, 0.5)
  psi <- as.vector(Phi %*% w)
  list(Phi = Phi, w = w, psi = psi,
       y = rbinom(n, 1, stats::plogis(psi)), n = n)
}

test_that("polyagamma is now a valid kernel and gibbs_step runs", {
  tbl <- valid_kernels()
  row <- tbl[tbl$posterior == "polyagamma" & tbl$features == "frozen", ]
  expect_true(row$valid)
  expect_match(row$reason, "Polya-Gamma")

  case <- pg_case(n = 100)
  mod <- bllnn_sampler(case$Phi, tau2 = 4, posterior = "polyagamma")
  expect_true(is_valid_kernel(mod))

  set_response(mod, case$y)
  # No sigma: the logistic link has no noise variance for the host to own.
  expect_no_error(f <- gibbs_step(mod))
  expect_length(f, case$n)
  expect_true(all(is.finite(f)))
})

test_that("the posterior mean lands on the maximum likelihood fit", {
  # The strongest check available: glm() is an entirely separate
  # implementation of logistic regression. With a diffuse prior the posterior
  # mean should sit on the MLE, and if the augmentation were wrong in any way
  # that mattered it would not.
  skip_on_cran()
  case <- pg_case(n = 400)

  mod <- bllnn_sampler(case$Phi, tau2 = 100, posterior = "polyagamma")
  set_response(mod, case$y)

  n_draws <- 2000
  W <- matrix(NA_real_, n_draws, ncol(case$Phi))
  set.seed(2)
  for (i in seq_len(n_draws)) {
    gibbs_step(mod)
    W[i, ] <- mod$w
  }
  post_mean <- colMeans(W[-(1:500), , drop = FALSE])

  mle <- unname(coef(stats::glm(case$y ~ case$Phi[, -1],
                                family = stats::binomial())))

  cat(sprintf("\n[polyagamma] posterior %s\n             glm      %s\n",
              paste(sprintf("%6.3f", post_mean), collapse = " "),
              paste(sprintf("%6.3f", mle), collapse = " ")))

  expect_equal(post_mean, mle, tolerance = 0.08)
})

test_that("the latent variables have the right conditional mean", {
  # omega_i | w ~ PG(1, psi_i), so the drawn omegas must track pg_mean(psi).
  # This is the augmentation step itself, checked separately from the draw of
  # w that consumes it.
  case <- pg_case(n = 300)
  mod <- bllnn_sampler(case$Phi, tau2 = 4, posterior = "polyagamma")
  set_response(mod, case$y)
  mod$w <- case$w                      # condition on known weights

  set.seed(3)
  acc <- numeric(case$n)
  reps <- 300
  for (i in seq_len(reps)) {
    mod$w <- case$w                    # hold w fixed across replicates
    gibbs_step(mod)
    acc <- acc + mod$omega
  }

  # Judged in Monte Carlo standard errors rather than on a relative scale.
  # These means sit near 0.2 with a per-draw sd near 0.19, so at 300
  # replicates the noise alone is about 6% -- a relative tolerance tight
  # enough to be meaningful would fail on noise, and one loose enough to pass
  # would not test anything.
  se <- sqrt(pg_var(case$psi) / reps)
  expect_lt(max(abs(acc / reps - pg_mean(case$psi)) / se), 4)
})

test_that("an offset shifts the fit and defaults to zero", {
  case <- pg_case(n = 200)

  mod <- bllnn_sampler(case$Phi, tau2 = 4, posterior = "polyagamma")
  expect_equal(mod$offset, rep(0, case$n))

  set_response(mod, case$y)
  set.seed(4); plain <- gibbs_step(mod)

  set_offset(mod, 2)
  expect_equal(mod$offset, rep(2, case$n))
  set.seed(4); shifted <- gibbs_step(mod)

  # Same seed, so any difference comes from the offset.
  expect_false(isTRUE(all.equal(plain, shifted)))

  # A large positive offset already explains the ones, so the features have
  # less work to do and the fitted part shifts down.
  expect_lt(mean(shifted), mean(plain))
})

test_that("the logistic path validates its inputs", {
  case <- pg_case(n = 100)
  mod <- bllnn_sampler(case$Phi, tau2 = 4, posterior = "polyagamma")

  expect_error(set_response(mod, rnorm(case$n)), "must be 0/1")
  expect_error(set_response(mod, rep(2, case$n)), "must be 0/1")
  expect_no_error(set_response(mod, case$y))

  expect_error(set_offset(mod, rnorm(case$n + 1)), "length 101")
  expect_error(set_offset(mod, "a"), "numeric")
  expect_error(set_offset(mod, c(1, NA)), "no NAs")
  expect_error(set_offset(42, 1), "bllnn_sampler")
})

test_that("the conjugate path still refuses to run without sigma", {
  # The logistic path does not need sigma, but relaxing that must not have
  # relaxed it for the Gaussian path, where the host owning sigma is rule 2.
  case <- pg_case(n = 80)
  mod <- bllnn_sampler(case$Phi, tau2 = 4, posterior = "conjugate")
  set_response(mod, rnorm(case$n))
  expect_error(gibbs_step(mod), "set_sigma")
})

test_that("tau2 auto calibrates on the latent scale for binary outcomes", {
  # var(y) for 0/1 data carries no scale information, so the calibration must
  # not use it.
  case <- pg_case(n = 200)
  mod <- bllnn_sampler(case$Phi, tau2 = "auto", posterior = "polyagamma")
  set_response(mod, case$y)

  expect_equal(mod$tau2, 1 / mean(rowSums(case$Phi^2)))
  expect_true(is.finite(mod$tau2) && mod$tau2 > 0)
  expect_no_error(gibbs_step(mod))
})
