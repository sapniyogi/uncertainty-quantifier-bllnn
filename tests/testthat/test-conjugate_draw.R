# --- shared helpers ---------------------------------------------------------

n_draws_mc <- 50000

# Monte Carlo standard errors for N iid Gaussian draws:
#   se(mean_j) = sqrt(V_jj / N)
#   se(cov_ij) = sqrt((V_ii V_jj + V_ij^2) / N)
mc_se_mean <- function(V, n_draws) sqrt(diag(V) / n_draws)

mc_se_cov <- function(V, n_draws) {
  m <- ncol(V)
  outer(seq_len(m), seq_len(m), Vectorize(function(i, j) {
    sqrt((V[i, i] * V[j, j] + V[i, j]^2) / n_draws)
  }))
}

# Every assertion below is stated as a multiple of those standard errors rather
# than as an absolute number, so the bound is set by sampling theory and not by
# what happens to pass. Five was chosen before any result was looked at: with
# m = 20 the covariance has 210 distinct entries, and the largest of K
# standardised deviations grows like sqrt(2 log 2K), which is about 3.5 for
# K = 210. Five leaves headroom for that multiplicity while still being tight
# enough to catch a genuinely wrong covariance.
k_se <- 5

draw_matrix <- function(Phi, r, sigma, tau2, n_draws, seed = 1) {
  set.seed(seed)
  out <- matrix(NA_real_, nrow = n_draws, ncol = ncol(Phi))
  for (i in seq_len(n_draws)) {
    out[i, ] <- conjugate_draw(Phi, r, sigma, tau2)
  }
  out
}

# Fraction of draws outside the nominal 95% Mahalanobis contour.
#
# KNOWN LIMITATION, recorded deliberately. conjugate_draw() forms
# w = mu + t(R) z with V = R'R, so evaluating the Mahalanobis distance against
# that same V gives
#   d^2 = (w - mu)' V^-1 (w - mu) = z' R (R'R)^-1 R' z = z'z
# identically, whatever mu and V happen to be. This statistic therefore
# verifies the normal generator and the Cholesky orientation -- using R where
# t(R) belongs breaks the cancellation -- and says NOTHING about whether the
# posterior moments are the right ones. The brute-force integration and
# dual-form tests are what check the moments. Kept because the generator is
# still worth testing, not because it validates the formula.
mahalanobis_tail <- function(draws, mu, V) {
  d2 <- stats::mahalanobis(draws, center = mu, cov = V)
  mean(d2 > stats::qchisq(0.95, df = length(mu)))
}

# Posterior moments via the n x n dual (kernel) form. Algebraically identical
# to the primal m x m form by the Woodbury identity, but it inverts a
# different matrix of a different size, so agreement is real evidence rather
# than a restatement.
dual_moments <- function(Phi, r, sigma, tau2) {
  n <- nrow(Phi)
  m <- ncol(Phi)
  K <- sigma^2 * diag(n) + tau2 * Phi %*% t(Phi)
  K_inv <- solve(K)
  list(
    mean = as.vector(tau2 * t(Phi) %*% K_inv %*% r),
    cov = tau2 * diag(m) - tau2^2 * t(Phi) %*% K_inv %*% Phi
  )
}

# How far solve() strayed: A V should be the identity.
solve_residual <- function(post) {
  max(abs(post$precision %*% post$cov - diag(ncol(post$cov))))
}

# Shared assertion block, so the 50,000-draw and 5,000-draw variants of each
# check cannot drift apart.
expect_draws_match_posterior <- function(Phi, r, sigma, tau2, n_draws, label,
                                         seed = 1) {
  post <- conjugate_moments(Phi, r, sigma, tau2)
  draws <- draw_matrix(Phi, r, sigma, tau2, n_draws, seed = seed)

  se_mean <- mc_se_mean(post$cov, n_draws)
  se_cov <- mc_se_cov(post$cov, n_draws)
  empirical_cov <- cov(draws)
  tail_frac <- mahalanobis_tail(draws, post$mean, post$cov)
  residual <- solve_residual(post)

  cat(sprintf(
    "\n[%s N=%d] kappa(Phi'Phi)=%.4g kappa(A)=%.4g |AV-I|=%.3g\n",
    label, n_draws, kappa(crossprod(Phi), exact = TRUE),
    kappa(post$precision, exact = TRUE), residual))
  cat(sprintf(
    "   max|cov err|=%.4g (%.2f se)  mahalanobis=%.4f\n",
    max(abs(empirical_cov - post$cov)),
    max(abs(empirical_cov - post$cov) / se_cov), tail_frac))

  expect_true(all(abs(colMeans(draws) - post$mean) < k_se * se_mean))
  expect_true(all(abs(empirical_cov - post$cov) < k_se * se_cov))
  expect_lt(residual, 1e-8)
  expect_equal(tail_frac, 0.05,
               tolerance = k_se * sqrt(0.05 * 0.95 / n_draws) / 0.05)

  invisible(post)
}

# n = 200, m = 20, every column the same latent vector plus independent noise.
collinear_phi <- function(n = 200, m = 20, noise_sd = 0.002, seed = 99) {
  set.seed(seed)
  latent <- rnorm(n)
  Phi <- matrix(rep(latent, m), nrow = n, ncol = m) +
    matrix(rnorm(n * m, sd = noise_sd), nrow = n, ncol = m)
  colnames(Phi) <- paste0("f", seq_len(m))
  Phi
}

well_conditioned_case <- function() {
  set.seed(20260822)
  Phi <- matrix(rnorm(40 * 3), 40, 3)
  list(Phi = Phi,
       r = as.vector(Phi %*% c(1, -2, 0.5)) + rnorm(40, sd = 0.7),
       sigma = 0.7, tau2 = 2)
}

collinear_case <- function(tau2 = 1) {
  Phi <- collinear_phi()
  set.seed(7)
  list(Phi = Phi,
       r = as.vector(Phi %*% rnorm(ncol(Phi))) + rnorm(nrow(Phi), sd = 1),
       sigma = 1, tau2 = tau2)
}

# Draw counts. The slow variants are the real diagnostics; the fast ones run
# everywhere so a regression cannot hide behind a skip. At 5,000 draws the
# standard errors are sqrt(10) wider, so the fast variants are genuinely
# weaker -- they are a tripwire, not a replacement.
n_draws_fast <- 5000

# --- 1. tiny hand-checkable case -------------------------------------------

test_that("posterior moments match the closed form on a tiny case (n=5, m=2)", {
  Phi <- matrix(c(1, 2, 3, 4, 5,
                  1, 0, 1, 0, 1), nrow = 5, ncol = 2)
  r <- c(2.1, 3.9, 6.2, 7.8, 10.1)
  sigma <- 0.5
  tau2 <- 4

  got <- conjugate_moments(Phi, r, sigma, tau2)

  # (a) Independent matrix-level closed form, written with `t(X) %*% X` rather
  # than the crossprod() the implementation uses.
  A <- (t(Phi) %*% Phi) / sigma^2 + diag(2) / tau2
  V <- solve(A)
  mu <- as.vector(V %*% t(Phi) %*% r / sigma^2)

  expect_equal(got$cov, V)
  expect_equal(got$mean, mu)
  expect_equal(got$precision, A)

  # (b) Fully scalar 2x2 inversion by the adjugate formula, reusing no matrix
  # routine at all. If both (a) and (b) agree with the implementation, the
  # result is not an artefact of any single solver.
  s2 <- sigma^2
  a11 <- sum(Phi[, 1]^2) / s2 + 1 / tau2
  a12 <- sum(Phi[, 1] * Phi[, 2]) / s2
  a22 <- sum(Phi[, 2]^2) / s2 + 1 / tau2
  det_a <- a11 * a22 - a12^2

  v11 <- a22 / det_a
  v12 <- -a12 / det_a
  v22 <- a11 / det_a

  b1 <- sum(Phi[, 1] * r) / s2
  b2 <- sum(Phi[, 2] * r) / s2

  expect_equal(got$cov,
               matrix(c(v11, v12, v12, v22), 2, 2))
  expect_equal(got$mean,
               c(v11 * b1 + v12 * b2,
                 v12 * b1 + v22 * b2))

  # The covariance must be exactly symmetric, not merely close, or chol() in
  # the draw is entitled to refuse it.
  expect_identical(got$cov, t(got$cov))

  w <- conjugate_draw(Phi, r, sigma, tau2)
  expect_length(w, 2)
  expect_true(all(is.finite(w)))
})

# --- breaking the circularity ----------------------------------------------
# Everything below the Monte Carlo line compares draws against
# conjugate_moments()'s own output, so a wrong FORMULA would satisfy all of it.
# These three tests derive the posterior by routes that never touch the
# conjugate identity.

test_that("brute-force numerical integration reproduces the posterior moments", {
  # Correlated columns on purpose: with a near-diagonal posterior an error in
  # the off-diagonal term would have nowhere to show up.
  set.seed(4)
  n <- 6
  phi1 <- rnorm(n)
  Phi <- cbind(phi1, 0.85 * phi1 + rnorm(n, sd = 0.5))
  r <- as.vector(Phi %*% c(0.8, -1.3)) + rnorm(n, sd = 0.4)
  sigma <- 0.8
  tau2 <- 1.5

  post <- conjugate_moments(Phi, r, sigma, tau2)

  # Grid over +/- 6 analytic sd in each direction. Placing the grid with the
  # analytic moments does not make this circular: a wrong mean would put the
  # grid in the wrong place and the numerical moments would then disagree with
  # it, which is a failure, not a pass.
  n_grid <- 400
  span <- 6
  sd1 <- sqrt(post$cov[1, 1])
  sd2 <- sqrt(post$cov[2, 2])
  g1 <- seq(post$mean[1] - span * sd1, post$mean[1] + span * sd1,
            length.out = n_grid)
  g2 <- seq(post$mean[2] - span * sd2, post$mean[2] + span * sd2,
            length.out = n_grid)
  w1 <- rep(g1, times = n_grid)
  w2 <- rep(g2, each = n_grid)

  # Unnormalised log posterior, evaluated directly as Gaussian likelihood plus
  # Gaussian prior. No conjugacy, no solve(), no chol() -- just the residual
  # sum of squares accumulated one observation at a time.
  ss <- numeric(length(w1))
  for (i in seq_len(n)) {
    ss <- ss + (r[i] - (Phi[i, 1] * w1 + Phi[i, 2] * w2))^2
  }
  log_post <- -ss / (2 * sigma^2) - (w1^2 + w2^2) / (2 * tau2)

  p <- exp(log_post - max(log_post))
  p <- p / sum(p)

  num_mean <- c(sum(p * w1), sum(p * w2))
  d1 <- w1 - num_mean[1]
  d2 <- w2 - num_mean[2]
  num_cov <- matrix(c(sum(p * d1 * d1), sum(p * d1 * d2),
                      sum(p * d1 * d2), sum(p * d2 * d2)), 2, 2)

  cat(sprintf(
    "\n[brute force] grid %dx%d (%d pts) over +/-%d sd\n",
    n_grid, n_grid, n_grid^2, span))
  cat(sprintf(
    "   rel mean dev=%.3e  rel cov dev=%.3e  cor: analytic %.6f numeric %.6f\n",
    max(abs(num_mean - post$mean) / abs(post$mean)),
    max(abs(num_cov - post$cov) / abs(post$cov)),
    stats::cov2cor(post$cov)[1, 2], stats::cov2cor(num_cov)[1, 2]))

  # Four significant figures. Accuracy here is limited by truncating the grid
  # at +/-6 sd, not by its resolution -- the deviation is flat from 200x200 to
  # 600x600 and moves by two orders of magnitude between +/-5 and +/-7 sd. The
  # achieved agreement is ~1e-7, three decades inside this bound, and a 1%
  # error in sigma moves it to ~1e-2, so the bound discriminates.
  expect_equal(num_mean, unname(post$mean), tolerance = 1e-4)
  expect_equal(num_cov, unname(post$cov), tolerance = 1e-4)
})

test_that("the dual form agrees with the primal form", {
  cases <- list(
    list(label = "tiny n=6 m=2", seed = 4, kind = "tiny",
         sigma = 0.8, tau2 = 1.5, tol = 1e-10),
    list(label = "well-conditioned m=3", kind = "well",
         sigma = 0.7, tau2 = 2, tol = 1e-10),
    list(label = "collinear tau2=1", kind = "collinear",
         sigma = 1, tau2 = 1, tol = 1e-8),
    # Both forms are near their numerical limits here: the dual form inverts an
    # n x n matrix scaled by tau2 = 1e6, so its accuracy degrades faster than
    # the primal. Asserted loosely and reported, rather than dropped.
    list(label = "collinear tau2=1e6", kind = "collinear",
         sigma = 1, tau2 = 1e6, tol = 1e-4),
    list(label = "m > n (3x7)", kind = "wide",
         sigma = 1, tau2 = 1, tol = 1e-10)
  )

  for (case in cases) {
    if (case$kind == "tiny") {
      set.seed(4)
      phi1 <- rnorm(6)
      Phi <- cbind(phi1, 0.85 * phi1 + rnorm(6, sd = 0.5))
      r <- as.vector(Phi %*% c(0.8, -1.3)) + rnorm(6, sd = 0.4)
    } else if (case$kind == "well") {
      set.seed(20260822)
      Phi <- matrix(rnorm(40 * 3), 40, 3)
      r <- as.vector(Phi %*% c(1, -2, 0.5)) + rnorm(40, sd = 0.7)
    } else if (case$kind == "collinear") {
      Phi <- collinear_phi()
      set.seed(7)
      r <- as.vector(Phi %*% rnorm(ncol(Phi))) + rnorm(nrow(Phi), sd = 1)
    } else {
      set.seed(11)
      Phi <- matrix(rnorm(3 * 7), 3, 7)
      r <- rnorm(3)
    }

    primal <- conjugate_moments(Phi, r, case$sigma, case$tau2)
    dual <- dual_moments(Phi, r, case$sigma, case$tau2)

    mean_abs <- max(abs(dual$mean - primal$mean))
    cov_abs <- max(abs(dual$cov - primal$cov))
    mean_rel <- mean_abs / max(abs(primal$mean))
    cov_rel <- cov_abs / max(abs(primal$cov))

    cat(sprintf(
      "[dual] %-22s mean abs=%.3e rel=%.3e | cov abs=%.3e rel=%.3e\n",
      case$label, mean_abs, mean_rel, cov_abs, cov_rel))

    expect_lt(mean_rel, case$tol)
    expect_lt(cov_rel, case$tol)
  }
})

test_that("solve() returns a genuine inverse in every test configuration", {
  cases <- list(
    list(label = "tiny n=5 m=2",
         Phi = matrix(c(1, 2, 3, 4, 5, 1, 0, 1, 0, 1), 5, 2),
         r = c(2.1, 3.9, 6.2, 7.8, 10.1), sigma = 0.5, tau2 = 4)
  )

  set.seed(20260822)
  Phi_w <- matrix(rnorm(40 * 3), 40, 3)
  cases[[2]] <- list(
    label = "well-conditioned m=3", Phi = Phi_w,
    r = as.vector(Phi_w %*% c(1, -2, 0.5)) + rnorm(40, sd = 0.7),
    sigma = 0.7, tau2 = 2)

  Phi_c <- collinear_phi()
  set.seed(7)
  r_c <- as.vector(Phi_c %*% rnorm(ncol(Phi_c))) + rnorm(nrow(Phi_c), sd = 1)
  cases[[3]] <- list(label = "collinear tau2=1", Phi = Phi_c, r = r_c,
                     sigma = 1, tau2 = 1)
  cases[[4]] <- list(label = "collinear tau2=1e6", Phi = Phi_c, r = r_c,
                     sigma = 1, tau2 = 1e6)

  set.seed(11)
  cases[[5]] <- list(label = "m > n (3x7)", Phi = matrix(rnorm(3 * 7), 3, 7),
                     r = rnorm(3), sigma = 1, tau2 = 1)

  for (case in cases) {
    post <- conjugate_moments(case$Phi, case$r, case$sigma, case$tau2)
    residual <- solve_residual(post)
    cat(sprintf("[solve residual] %-22s |AV - I|_max = %.3e\n",
                case$label, residual))
    expect_lt(residual, 1e-8)
  }
})

# --- 2. Monte Carlo check ---------------------------------------------------

test_that("50,000 draws reproduce the analytic mean and full covariance", {
  skip_on_cran()
  case <- well_conditioned_case()
  post <- expect_draws_match_posterior(case$Phi, case$r, case$sigma, case$tau2,
                                       n_draws_mc, "well-conditioned m=3")

  # The full matrix, off-diagonals included: the correlation structure between
  # weights is the part a diagonal-only check would miss.
  draws <- draw_matrix(case$Phi, case$r, case$sigma, case$tau2, n_draws_mc)
  expect_equal(cov(draws), post$cov, tolerance = 0.02)
})

test_that("5,000 draws reproduce the analytic mean and full covariance", {
  case <- well_conditioned_case()
  expect_draws_match_posterior(case$Phi, case$r, case$sigma, case$tau2,
                               n_draws_fast, "well-conditioned m=3")
})

# A collinear design, where the posterior correlation is strong enough that
# ignoring it is unmistakable. This is the assertion form of the check that
# previously lived only in the title of an inst/validation plot.
expect_orientation_detected <- function(n_draws) {
  set.seed(5)
  n <- 60
  phi1 <- rnorm(n)
  Phi <- cbind(f1 = phi1, f2 = 0.97 * phi1 + rnorm(n, sd = 0.24))
  r <- as.vector(Phi %*% c(1.5, -0.5)) + rnorm(n, sd = 0.5)
  sigma <- 0.5
  tau2 <- 10

  post <- conjugate_moments(Phi, r, sigma, tau2)
  analytic_cor <- stats::cov2cor(post$cov)[1, 2]

  draws <- draw_matrix(Phi, r, sigma, tau2, n_draws, seed = 2)
  empirical_cor <- cor(draws[, 1], draws[, 2])

  # The failure mode: exact marginal variances, drawn independently.
  set.seed(3)
  draws_diag <- cbind(
    rnorm(n_draws, post$mean[1], sqrt(post$cov[1, 1])),
    rnorm(n_draws, post$mean[2], sqrt(post$cov[2, 2]))
  )
  diag_cor <- cor(draws_diag[, 1], draws_diag[, 2])

  cat(sprintf(
    "[orientation N=%d] analytic cor=%.4f  conjugate_draw=%.4f  diagonal-only=%.4f\n",
    n_draws, analytic_cor, empirical_cor, diag_cor))

  # se(r) ~ (1 - r^2) / sqrt(N); se(sd) / sd ~ 1 / sqrt(2N). Both bounds are
  # k_se multiples of those, so they tighten with N rather than being fixed.
  expect_lt(abs(empirical_cor - analytic_cor),
            k_se * (1 - analytic_cor^2) / sqrt(n_draws))

  # The diagonal-only version does not reproduce it. Without this second
  # assertion the first proves nothing about the sensitivity of the check.
  expect_gt(abs(diag_cor - analytic_cor), 0.5)

  # And it is a genuinely sneaky bug: the marginals are indistinguishable, so
  # only the joint check separates the two.
  analytic_sd <- unname(sqrt(diag(post$cov)))
  sd_tol <- k_se / sqrt(2 * n_draws)
  expect_equal(unname(apply(draws_diag, 2, sd)), analytic_sd, tolerance = sd_tol)
  expect_equal(unname(apply(draws, 2, sd)), analytic_sd, tolerance = sd_tol)
}

test_that("the diagonal-only diagnostic detects the bug it exists to catch", {
  skip_on_cran()
  expect_orientation_detected(n_draws_mc)
})

test_that("the diagonal-only diagnostic detects the bug at 5,000 draws", {
  expect_orientation_detected(n_draws_fast)
})

# --- 3. wide prior collapses to OLS ----------------------------------------

test_that("a wide prior drives the posterior mean to the OLS fit", {
  set.seed(3)
  n <- 200
  m <- 5
  Phi <- matrix(rnorm(n * m), n, m)
  r <- as.vector(Phi %*% c(2, -1, 0.5, 0, 3)) + rnorm(n, sd = 1)

  ols <- unname(lm.fit(Phi, r)$coefficients)
  post_mean <- conjugate_moments(Phi, r, sigma = 1, tau2 = 1e6)$mean

  expect_equal(post_mean, ols, tolerance = 1e-6)

  # The prior is a ridge penalty of size sigma^2 / tau2, so the gap to OLS
  # shrinks in proportion to 1 / tau2 rather than merely getting small.
  gap <- function(t2) {
    max(abs(conjugate_moments(Phi, r, sigma = 1, tau2 = t2)$mean - ols))
  }
  expect_equal(gap(1e4) / gap(1e6), 100, tolerance = 0.01)
})

# --- 4. reproducibility -----------------------------------------------------

test_that("the draw is reproducible under a fixed seed", {
  set.seed(3)
  Phi <- matrix(rnorm(50 * 3), 50, 3)
  r <- rnorm(50)

  set.seed(7)
  a <- conjugate_draw(Phi, r, sigma = 1, tau2 = 5)
  set.seed(7)
  b <- conjugate_draw(Phi, r, sigma = 1, tau2 = 5)
  set.seed(8)
  d <- conjugate_draw(Phi, r, sigma = 1, tau2 = 5)

  expect_identical(a, b)
  expect_false(isTRUE(all.equal(a, d)))

  # Consecutive draws from one stream differ too, so the draw is consuming
  # randomness rather than returning the posterior mean every time.
  set.seed(7)
  two <- replicate(2, conjugate_draw(Phi, r, sigma = 1, tau2 = 5))
  expect_false(isTRUE(all.equal(two[, 1], two[, 2])))
})

# --- stress tests: near-collinear features ---------------------------------
# Diagnostics, not regression tests. They ask whether the naive solve()/chol()
# still produce honest draws when the cross-product is badly conditioned. If
# one fails, that is a finding about the linear algebra, not a licence to
# loosen the bound.

test_that("50,000 draws stay exact under near-collinear features", {
  skip_on_cran()
  case <- collinear_case(tau2 = 1)
  expect_gt(kappa(crossprod(case$Phi), exact = TRUE), 1e6)
  expect_draws_match_posterior(case$Phi, case$r, case$sigma, case$tau2,
                               n_draws_mc, "stress 1: tau2=1")
})

test_that("5,000 draws stay exact under near-collinear features", {
  case <- collinear_case(tau2 = 1)
  expect_gt(kappa(crossprod(case$Phi), exact = TRUE), 1e6)
  expect_draws_match_posterior(case$Phi, case$r, case$sigma, case$tau2,
                               n_draws_fast, "stress 1: tau2=1")
})

test_that("50,000 draws stay exact under collinear features with a wide prior", {
  # The worst realistic case. At tau2 = 1e6 the I/tau2 term is too small to
  # regularise the cross-product, so the posterior precision inherits the full
  # conditioning of Phi'Phi rather than the prior's.
  skip_on_cran()
  case <- collinear_case(tau2 = 1e6)
  post <- conjugate_moments(case$Phi, case$r, case$sigma, case$tau2)

  # Confirms the premise: the prior is no longer doing the regularising.
  expect_gt(kappa(post$precision, exact = TRUE), 1e6)

  expect_draws_match_posterior(case$Phi, case$r, case$sigma, case$tau2,
                               n_draws_mc, "stress 2: tau2=1e6")
})

test_that("5,000 draws stay exact under collinear features with a wide prior", {
  case <- collinear_case(tau2 = 1e6)
  post <- conjugate_moments(case$Phi, case$r, case$sigma, case$tau2)
  expect_gt(kappa(post$precision, exact = TRUE), 1e6)
  expect_draws_match_posterior(case$Phi, case$r, case$sigma, case$tau2,
                               n_draws_fast, "stress 2: tau2=1e6")
})

# --- properties worth pinning ----------------------------------------------

test_that("the draw is proper when m > n", {
  # Ordinary least squares is undefined here; the proper prior is what makes
  # the posterior well defined.
  set.seed(11)
  Phi <- matrix(rnorm(3 * 7), nrow = 3, ncol = 7)
  r <- rnorm(3)

  w <- conjugate_draw(Phi, r, sigma = 1, tau2 = 1)

  expect_length(w, 7)
  expect_true(all(is.finite(w)))
})

test_that("column names of Phi are carried through", {
  Phi <- matrix(c(1, 2, 3, 4), 2, 2, dimnames = list(NULL, c("f1", "f2")))
  w <- conjugate_draw(Phi, c(1, 2), sigma = 1, tau2 = 1)

  expect_named(w, c("f1", "f2"))
})

test_that("invalid arguments are refused", {
  Phi <- matrix(rnorm(20), 10, 2)
  r <- rnorm(10)

  expect_error(conjugate_draw(as.data.frame(Phi), r, 1, 1), "numeric matrix")
  expect_error(conjugate_draw(Phi[, 1], r, 1, 1), "numeric matrix")
  expect_error(conjugate_draw(Phi, rnorm(9), 1, 1), "length 9")
  expect_error(conjugate_draw(Phi, r, 0, 1), "positive")
  expect_error(conjugate_draw(Phi, r, -1, 1), "positive")
  expect_error(conjugate_draw(Phi, r, 1, 0), "positive")
  expect_error(conjugate_draw(Phi, r, c(1, 2), 1), "single positive")

  Phi_na <- Phi
  Phi_na[1, 1] <- NA
  expect_error(conjugate_draw(Phi_na, r, 1, 1), "must not contain NA")
  expect_error(conjugate_draw(Phi, replace(r, 1, NA), 1, 1), "no NAs")
})
