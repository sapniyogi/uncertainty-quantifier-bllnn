fit_case <- function(n = 200, seed = 1) {
  sim <- sim_partial_linear(n = n, beta = c(treat = 1.5), p_z = 5,
                            f = "friedman", confounding = 0.6, sigma = 1,
                            seed = seed)
  sim
}

quick_fit <- function(sim = fit_case(), ...) {
  bllnn(y ~ z1 + z2 + z3 + z4 + z5, data = sim$data, linear = ~ treat,
        folds = 3, width = 6, epochs = 120, n_iter = 250, burn = 80,
        seed = 1, ...)
}

# --- the whole reason this function exists ----------------------------------

test_that("the wrapper assembles the configuration the gate validated", {
  sim <- fit_case()
  fit <- quick_fit(sim)

  # 1. cross-fitted bodies
  expect_s3_class(fit$crossfit, "bllnn_crossfit")
  expect_equal(fit$crossfit$n_folds, 3L)

  # 2. the confounding channel reached every fold
  expect_true(fit$crossfit$has_linear)
  expect_true(any(grepl("ehat_treat", colnames(feature_matrix(fit$crossfit)))))

  # 3. the design the host regressed on is residualised, not raw. This is the
  # step whose absence cost half the remaining bias, and it is invisible in
  # the output when it is missing.
  expect_equal(fit$X_used, partial_out(fit$crossfit))
  expect_false(isTRUE(all.equal(unname(fit$X_used), unname(fit$X))))

  # 4. tau2 came from the data rather than a guess
  expect_equal(fit$tau2_mode, "auto")
  expect_true(is.finite(fit$tau2) && fit$tau2 > 0)
})

test_that("the estimate is closer to the truth than naive least squares", {
  # Averaged over several datasets rather than one. The confounding that OLS
  # picks up varies a lot between draws -- across seeds it ranges from below
  # the truth to well above it -- so a single dataset would make this a test
  # of which seed was chosen. The real claim lives in the 100-dataset coverage
  # simulation; this is the smoke check that the wrapper wires it up.
  skip_on_cran()
  beta_true <- 1.5

  res <- vapply(1:4, function(i) {
    sim <- fit_case(n = 400, seed = i)
    fit <- bllnn(y ~ z1 + z2 + z3 + z4 + z5, data = sim$data,
                 linear = ~ treat, folds = 5, width = 20, epochs = 600,
                 n_iter = 900, burn = 300, seed = i)
    c(bllnn = unname(coef(fit)[["treat"]]),
      ols = unname(coef(lm(y ~ treat, data = sim$data))[2]))
  }, numeric(2))

  bias <- rowMeans(res) - beta_true
  cat(sprintf("\n[bllnn] over 4 datasets: bllnn bias %+.4f, naive OLS %+.4f\n",
              bias[["bllnn"]], bias[["ols"]]))

  expect_lt(abs(bias[["bllnn"]]), abs(bias[["ols"]]))
  # OLS carries the confounding; the wrapper must remove most of it.
  expect_lt(abs(bias[["bllnn"]]), 0.5 * abs(bias[["ols"]]))
})

# --- the formula interface --------------------------------------------------

test_that("formula and linear terms are parsed into the right matrices", {
  sim <- fit_case(n = 150)
  fit <- quick_fit(sim)

  expect_equal(colnames(fit$Z), paste0("z", 1:5))
  expect_equal(colnames(fit$X), "treat")
  # No intercept in either: the network carries the level, and an intercept in
  # X would be collinear with it.
  expect_false("(Intercept)" %in% colnames(fit$Z))
  expect_false("(Intercept)" %in% colnames(fit$X))
  expect_equal(fit$n, 150L)
})

test_that("rows with missing values are dropped consistently across parts", {
  sim <- fit_case(n = 150)
  df <- sim$data
  df$z2[c(3, 10)] <- NA
  df$treat[20] <- NA

  fit <- bllnn(y ~ z1 + z2 + z3 + z4 + z5, data = df, linear = ~ treat,
               folds = 2, width = 5, epochs = 80, n_iter = 150, burn = 50,
               seed = 1)

  # One model frame for both parts, so y, Z and X cannot fall out of step.
  expect_equal(fit$n, 147L)
  expect_equal(fit$n_dropped, 3L)
  expect_equal(nrow(fit$Z), 147L)
  expect_equal(nrow(fit$X), 147L)
  expect_length(fit$y, 147L)
})

test_that("a model with no linear part still fits", {
  sim <- fit_case(n = 150)
  fit <- bllnn(y ~ z1 + z2 + z3 + z4 + z5, data = sim$data, linear = NULL,
               folds = 2, width = 5, epochs = 80, n_iter = 150, burn = 50,
               seed = 1)

  expect_equal(ncol(fit$beta), 0L)
  expect_length(coef(fit), 0L)
  expect_error(confint(fit), "no linear terms")
  expect_length(fit$f_mean, 150L)
  expect_output(print(fit), "linear terms : none")
})

test_that("a fixed seed reproduces the whole fit", {
  sim <- fit_case(n = 150)
  a <- quick_fit(sim)
  b <- quick_fit(sim)

  expect_equal(a$beta, b$beta)
  expect_equal(a$sigma2, b$sigma2)
  expect_equal(a$f_mean, b$f_mean)

  set.seed(99); before <- runif(3)
  set.seed(99); invisible(quick_fit(sim)); after <- runif(3)
  expect_equal(before, after)
})

test_that("arguments are validated", {
  sim <- fit_case(n = 100)

  expect_error(bllnn(~ z1, data = sim$data), "two-sided")
  expect_error(bllnn(y ~ z1, data = as.matrix(sim$data)), "data frame")
  expect_error(bllnn(y ~ z1, data = sim$data, n_iter = 100, burn = 100),
               "nothing would be kept")
  expect_error(bllnn(y ~ z1, data = sim$data, n_iter = 0), "positive integer")
  expect_error(bllnn(y ~ z1, data = sim$data, linear = treat ~ z1),
               "one-sided")
  expect_error(bllnn(y ~ z1, data = sim$data, keep_f = NA), "TRUE or FALSE")

  df <- sim$data
  df$y <- factor(rep(c("a", "b"), 50))
  expect_error(bllnn(y ~ z1 + z2, data = df), "must be numeric")
})

# --- the methods ------------------------------------------------------------

test_that("coef and confint agree with the stored draws", {
  fit <- quick_fit()

  expect_equal(coef(fit), colMeans(fit$beta))
  ci <- confint(fit)
  expect_equal(unname(ci[1, ]),
               unname(quantile(fit$beta[, 1], c(0.025, 0.975))))

  ci90 <- confint(fit, level = 0.90)
  # A narrower level must give a narrower interval.
  expect_lt(ci90[1, 2] - ci90[1, 1], ci[1, 2] - ci[1, 1])

  expect_error(confint(fit, parm = "nonesuch"), "Unknown coefficient")
  expect_error(confint(fit, level = 1), "between 0 and 1")
})

test_that("summary reports the interval, the ESS, and warns when mixing is poor", {
  fit <- quick_fit()
  s <- summary(fit)

  expect_s3_class(s, "summary.bllnn_fit")
  expect_equal(rownames(s$table), "treat")
  expect_true(is.finite(s$table$ess))
  expect_equal(s$table$mean, unname(coef(fit)))

  out <- capture.output(print(s))
  expect_true(any(grepl("credible intervals", out)))
  expect_true(any(grepl("smallest ESS", out)))
  expect_true(any(grepl("prior variance", out)))

  # The warning must fire on a chain that has not mixed, or it is decoration.
  bad <- fit
  bad$beta <- matrix(cumsum(rnorm(200)), ncol = 1,
                     dimnames = list(NULL, "treat"))
  expect_true(any(grepl("has not mixed", capture.output(print(summary(bad))))))
})

test_that("predict returns fitted values and honest intervals", {
  fit <- quick_fit()

  f_only <- predict(fit, type = "f")
  resp <- predict(fit, type = "response")
  expect_length(f_only, fit$n)
  expect_equal(f_only, fit$f_mean)
  # The response adds the linear part, so the two must differ.
  expect_false(isTRUE(all.equal(f_only, resp)))
  expect_equal(resp, fit$f_mean + as.vector(fit$X_used %*% coef(fit)))

  band <- predict(fit, type = "f", interval = TRUE)
  expect_equal(colnames(band), c("fit", "lower", "upper"))
  expect_true(all(band[, "lower"] <= band[, "fit"]))
  expect_true(all(band[, "fit"] <= band[, "upper"]))
})

test_that("predict on new data averages the folds", {
  sim <- fit_case(n = 150)
  fit <- quick_fit(sim)
  nd <- sim$data[1:10, ]

  p <- predict(fit, newdata = nd, type = "f")
  expect_length(p, 10L)
  expect_true(all(is.finite(p)))

  # Reproduce the documented rule by hand: mean over folds of each body's
  # features times that fold's weight draws.
  cf <- fit$crossfit
  m_k <- cf$m_per_fold
  acc <- 0
  for (k in seq_len(cf$n_folds)) {
    Phi_k <- feature_matrix(cf$bodies[[k]], as.matrix(nd[, paste0("z", 1:5)]))
    cols <- seq_len(m_k) + (k - 1) * m_k
    acc <- acc + fit$f_weight_draws[, cols, drop = FALSE] %*% t(Phi_k)
  }
  expect_equal(p, colMeans(acc / cf$n_folds))
})

test_that("predict refuses what it cannot honestly do", {
  sim <- fit_case(n = 120)
  fit <- quick_fit(sim)

  expect_error(predict(fit, newdata = sim$data[1:5, ], type = "response"),
               "not yet implemented")
  expect_error(predict(fit, newdata = "nope"), "data frame")
  expect_error(predict(fit, interval = NA), "TRUE or FALSE")

  light <- quick_fit(sim, keep_f = FALSE)
  expect_null(light$f_draws)
  expect_error(predict(light, interval = TRUE), "keep_f = TRUE")
  # The point estimate still works without stored draws.
  expect_length(predict(light), light$n)
})

test_that("plot draws without error and refuses an empty request", {
  fit <- quick_fit()
  pdf(NULL)
  on.exit(dev.off(), add = TRUE)

  expect_invisible(plot(fit, which = "trace"))
  expect_no_error(plot(fit, which = c("trace", "density")))
  expect_no_error(plot(fit, which = "fitted"))

  sim <- fit_case(n = 120)
  no_linear <- bllnn(y ~ z1 + z2 + z3 + z4 + z5, data = sim$data,
                     folds = 2, width = 5, epochs = 80, n_iter = 150,
                     burn = 50, seed = 1)
  expect_error(plot(no_linear, which = "trace"), "Nothing to plot")
  expect_no_error(plot(no_linear, which = "fitted"))
})

test_that("print shows the call and the fitted interval", {
  fit <- quick_fit()
  out <- capture.output(print(fit))

  expect_true(any(grepl("bllnn_fit", out)))
  expect_true(any(grepl("Call:", out)))
  expect_true(any(grepl("treat", out)))
  expect_true(any(grepl("cross-fitting", out)))
})
