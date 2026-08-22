# The central invariant: the returned truth must reproduce y exactly. If this
# ever fails, every downstream test that scores an estimator against the truth
# is silently measuring the wrong thing.

test_that("the returned truth reconstructs y exactly", {
  sim <- sim_partial_linear(n = 100, seed = 42)

  expect_identical(sim$data$y, sim$mu_true + sim$noise)
  expect_identical(sim$mu_true, as.vector(sim$X %*% sim$beta) + sim$f_true)
})

test_that("structure and names are as documented", {
  sim <- sim_partial_linear(n = 30, beta = c(1, -0.5), p_z = 5, seed = 1)

  expect_named(sim$data, c("y", "x1", "x2", "z1", "z2", "z3", "z4", "z5"))
  expect_equal(nrow(sim$data), 30)
  expect_equal(dim(sim$X), c(30L, 2L))
  expect_equal(dim(sim$Z), c(30L, 5L))
  expect_equal(colnames(sim$X), c("x1", "x2"))
  expect_equal(colnames(sim$Z), paste0("z", 1:5))
  expect_length(sim$f_true, 30)
  expect_length(sim$mu_true, 30)
  expect_length(sim$noise, 30)
  expect_equal(sim$f_name, "friedman")
})

test_that("names on beta become the X column names", {
  sim <- sim_partial_linear(n = 20, beta = c(treat = 1.5, age = -0.2), seed = 1)

  expect_equal(colnames(sim$X), c("treat", "age"))
  expect_named(sim$beta, c("treat", "age"))
  expect_true(all(c("treat", "age") %in% names(sim$data)))
})

test_that("center_f controls the identification constant", {
  sim_c <- sim_partial_linear(n = 500, seed = 4, center_f = TRUE)
  sim_u <- sim_partial_linear(n = 500, seed = 4, center_f = FALSE)

  expect_equal(mean(sim_c$f_true), 0)
  # Friedman is strictly positive, so the uncentred version cannot be mean zero.
  expect_gt(mean(sim_u$f_true), 1)
  # Centring shifts f by a constant and nothing else.
  expect_equal(sim_u$f_true - mean(sim_u$f_true), sim_c$f_true)
})

test_that("sigma = 0 gives noiseless data", {
  sim <- sim_partial_linear(n = 50, sigma = 0, seed = 5)

  expect_true(all(sim$noise == 0))
  expect_identical(sim$data$y, sim$mu_true)
})

test_that("the same seed reproduces the same data", {
  a <- sim_partial_linear(n = 40, seed = 123)
  b <- sim_partial_linear(n = 40, seed = 123)
  c <- sim_partial_linear(n = 40, seed = 124)

  expect_identical(a$data, b$data)
  expect_identical(a$f_true, b$f_true)
  expect_false(isTRUE(all.equal(a$data$y, c$data$y)))
})

test_that("passing a seed does not disturb the random stream of the caller", {
  set.seed(99)
  before <- runif(3)

  set.seed(99)
  invisible(sim_partial_linear(n = 10, seed = 1))
  after <- runif(3)

  expect_equal(before, after)
})

# --- the confounding mechanism ---------------------------------------------
# This is what the simulator exists for. `confounding` is specified as roughly
# the correlation between each column of X and its nonlinear function of Z, so
# check that it actually reads that way, then check the consequence.

test_that("confounding reads as the correlation between X and m(Z)", {
  for (cf in c(0, 0.3, 0.6, 0.9)) {
    sim <- sim_partial_linear(n = 20000, beta = 1, confounding = cf, seed = 11)
    observed <- cor(sim$X[, 1], m_confounder(sim$Z, 1))
    expect_equal(observed, cf, tolerance = 0.03)
  }
})

test_that("naive OLS is unbiased when X and Z are independent", {
  sim <- sim_partial_linear(n = 4000, beta = c(1, -0.5),
                            confounding = 0, seed = 7)
  est <- coef(lm(y ~ x1 + x2, data = sim$data))[-1]

  expect_lt(max(abs(est - sim$beta)), 0.15)
})

test_that("naive OLS is biased when X and Z are dependent", {
  sim <- sim_partial_linear(n = 4000, beta = c(1, -0.5),
                            confounding = 0.9, seed = 7)
  est <- coef(lm(y ~ x1 + x2, data = sim$data))[-1]

  # Bias is large and, for x2, large enough to flip the sign of the effect.
  expect_gt(min(abs(est - sim$beta)), 0.3)
  expect_gt(est[["x2"]], 0)
})

test_that("the bias is attributable to f(Z) and nothing else", {
  # Given the true f(Z), OLS on the partialled-out response recovers beta even
  # under heavy confounding. This is the target any correct sampler must hit.
  sim <- sim_partial_linear(n = 4000, beta = c(1, -0.5),
                            confounding = 0.9, seed = 7)
  est <- coef(lm(sim$data$y - sim$f_true ~ sim$X - 1))

  expect_lt(max(abs(est - sim$beta)), 0.15)
})

test_that("f is genuinely nonlinear in Z", {
  for (fn in c("friedman", "sine")) {
    sim <- sim_partial_linear(n = 4000, f = fn, seed = 3)
    r2 <- summary(lm(sim$f_true ~ sim$Z))$r.squared
    expect_lt(r2, 0.95)
  }
})

test_that("a custom f is accepted and used", {
  my_f <- function(Z) Z[, 1]^3 - Z[, 2]
  sim <- sim_partial_linear(n = 60, p_z = 2, f = my_f, seed = 8)

  raw <- my_f(sim$Z)
  expect_equal(sim$f_true, raw - mean(raw))
  expect_equal(sim$f_name, "custom")
})

test_that("invalid arguments are refused", {
  expect_error(sim_partial_linear(n = 0), "positive integer")
  expect_error(sim_partial_linear(n = 2.5), "positive integer")
  expect_error(sim_partial_linear(beta = character(1)), "numeric vector")
  expect_error(sim_partial_linear(p_z = 0), "positive integer")
  expect_error(sim_partial_linear(sigma = -1), "non-negative")
  expect_error(sim_partial_linear(center_f = NA), "TRUE or FALSE")

  # confounding = 1 makes beta unidentified, so it is refused rather than run.
  expect_error(sim_partial_linear(confounding = 1), "not identified")
  expect_error(sim_partial_linear(confounding = -0.1), "\\[0, 1\\)")

  # friedman needs five columns of Z; sine does not.
  expect_error(sim_partial_linear(p_z = 4, f = "friedman"), "at least 5")
  expect_no_error(sim_partial_linear(n = 10, p_z = 1, f = "sine"))

  expect_error(sim_partial_linear(n = 10, f = function(Z) "not numeric"),
               "numeric vector of length")
  expect_error(sim_partial_linear(n = 10, f = function(Z) c(1, 2)),
               "numeric vector of length")
})
