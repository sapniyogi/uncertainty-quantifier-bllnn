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

# --- 2. Monte Carlo check ---------------------------------------------------

test_that("50,000 draws reproduce the analytic mean and full covariance", {
  set.seed(20260822)
  n <- 40
  m <- 3
  Phi <- matrix(rnorm(n * m), n, m)
  r <- as.vector(Phi %*% c(1, -2, 0.5)) + rnorm(n, sd = 0.7)
  sigma <- 0.7
  tau2 <- 2

  post <- conjugate_moments(Phi, r, sigma, tau2)

  n_draws <- 50000
  set.seed(1)
  draws <- matrix(NA_real_, nrow = n_draws, ncol = m)
  for (i in seq_len(n_draws)) {
    draws[i, ] <- conjugate_draw(Phi, r, sigma, tau2)
  }

  # Monte Carlo standard errors for a Gaussian sample:
  #   se(mean_j)  = sqrt(V_jj / N)
  #   se(cov_ij) ~= sqrt((V_ii V_jj + V_ij^2) / N)
  # Five standard errors is loose enough not to be flaky and tight enough that
  # a wrong covariance -- for instance V used where its Cholesky belongs, or a
  # transpose slip -- fails comfortably.
  se_mean <- sqrt(diag(post$cov) / n_draws)
  se_cov <- outer(seq_len(m), seq_len(m), Vectorize(function(i, j) {
    sqrt((post$cov[i, i] * post$cov[j, j] + post$cov[i, j]^2) / n_draws)
  }))

  expect_true(all(abs(colMeans(draws) - post$mean) < 5 * se_mean))

  # The full matrix, off-diagonals included: the correlation structure between
  # weights is the part a diagonal-only check would miss.
  empirical_cov <- cov(draws)
  expect_true(all(abs(empirical_cov - post$cov) < 5 * se_cov))
  expect_equal(empirical_cov, post$cov, tolerance = 0.02)
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
