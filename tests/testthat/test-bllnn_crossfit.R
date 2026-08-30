cf_case <- function(n = 300, p = 5) {
  sim <- sim_partial_linear(n = n, beta = c(treat = 1.5), p_z = p,
                            f = "friedman", confounding = 0.6, sigma = 1,
                            seed = 20260829)
  list(Z = sim$Z, y = sim$data$y, X = sim$X, sim = sim,
       f_true = sim$f_true)
}

# --- the sample-splitting guarantee ----------------------------------------

test_that("every row's features come from a body that never saw that row", {
  case <- cf_case(n = 200)
  cf <- bllnn_crossfit(case$Z, case$y, folds = 4, width = 8, epochs = 120,
                       seed = 1)

  expect_equal(cf$n_folds, 4L)
  expect_length(cf$fold, 200)
  expect_setequal(unique(cf$fold), 1:4)

  # Rebuild each fold's block from its own body and the rows it held out.
  # Agreement proves the assembly used the out-of-fold body throughout: if any
  # block had been filled from a body trained on those rows, the guarantee
  # would be broken and this would not match.
  Phi <- feature_matrix(cf)
  m_k <- cf$m_per_fold
  for (k in seq_len(cf$n_folds)) {
    held <- cf$fold == k
    cols <- seq_len(m_k) + (k - 1) * m_k
    expect_equal(unname(Phi[held, cols]),
                 unname(feature_matrix(cf$bodies[[k]],
                                       case$Z[held, , drop = FALSE])))
  }
})

test_that("the feature matrix is block diagonal by fold", {
  case <- cf_case(n = 150)
  cf <- bllnn_crossfit(case$Z, case$y, folds = 3, width = 6, epochs = 100,
                       seed = 1)
  Phi <- feature_matrix(cf)
  m_k <- cf$m_per_fold

  expect_equal(ncol(Phi), 3L * m_k)
  expect_equal(nrow(Phi), 150L)

  for (k in 1:3) {
    rows <- cf$fold == k
    own <- seq_len(m_k) + (k - 1) * m_k
    # Off-block entries must be exactly zero, not merely small.
    expect_true(all(Phi[rows, -own] == 0))
    # The intercept of its own block is present, so the block is not empty.
    expect_true(all(Phi[rows, own[1]] == 1))
  }
})

test_that("block diagonal columns are named by fold", {
  case <- cf_case(n = 120)
  cf <- bllnn_crossfit(case$Z, case$y, folds = 2, width = 4, epochs = 80,
                       seed = 1)
  nms <- colnames(feature_matrix(cf))

  expect_equal(nms[1], "f1_intercept")
  expect_equal(nms[2], "f1_h1")
  expect_equal(nms[cf$m_per_fold + 1], "f2_intercept")
  expect_length(unique(nms), length(nms))
})

test_that("cross-fitting uses the whole sample", {
  case <- cf_case(n = 200)
  cf <- bllnn_crossfit(case$Z, case$y, folds = 5, width = 6, epochs = 80,
                       seed = 1)

  expect_equal(nrow(feature_matrix(cf)), nrow(case$Z))
  # Every row is covered by exactly one block, so no row has an all-zero row.
  expect_true(all(rowSums(abs(feature_matrix(cf))) > 0))
})

# --- the block structure is bookkeeping, not an approximation --------------

test_that("a block-diagonal draw equals independent per-fold draws", {
  # The posterior factorises across blocks, so drawing from the assembled
  # system must give the same distribution as drawing each fold separately.
  # Checked through the precision matrix, which is exact rather than
  # statistical: off-block entries of Phi'Phi are identically zero.
  case <- cf_case(n = 150)
  cf <- bllnn_crossfit(case$Z, case$y, folds = 3, width = 5, epochs = 80,
                       seed = 1)
  Phi <- feature_matrix(cf)
  m_k <- cf$m_per_fold

  cross <- crossprod(Phi)
  for (k in 1:3) {
    own <- seq_len(m_k) + (k - 1) * m_k
    expect_true(all(cross[own, -own] == 0))
  }

  mod <- bllnn_sampler(cf, tau2 = 1)
  expect_equal(mod$m, 3L * m_k)
  expect_true(is_valid_kernel(mod))
})

# --- reproducibility and folds ---------------------------------------------

test_that("a fixed seed reproduces the split and the bodies", {
  case <- cf_case(n = 150)
  a <- bllnn_crossfit(case$Z, case$y, folds = 3, width = 5, epochs = 80, seed = 3)
  b <- bllnn_crossfit(case$Z, case$y, folds = 3, width = 5, epochs = 80, seed = 3)
  d <- bllnn_crossfit(case$Z, case$y, folds = 3, width = 5, epochs = 80, seed = 4)

  expect_equal(a$fold, b$fold)
  expect_equal(feature_matrix(a), feature_matrix(b))
  expect_false(isTRUE(all.equal(a$fold, d$fold)))

  set.seed(99); before <- runif(3)
  set.seed(99)
  invisible(bllnn_crossfit(case$Z, case$y, folds = 2, width = 4,
                           epochs = 50, seed = 1))
  after <- runif(3)
  expect_equal(before, after)
})

test_that("an explicit fold vector is honoured", {
  case <- cf_case(n = 120)
  my_fold <- rep(1:3, length.out = 120)
  cf <- bllnn_crossfit(case$Z, case$y, folds = my_fold, width = 4,
                       epochs = 60, seed = 1)

  expect_equal(cf$fold, as.integer(my_fold))
  expect_equal(cf$n_folds, 3L)
})

test_that("the confounding channel is carried into every fold", {
  case <- cf_case(n = 200)
  cf <- bllnn_crossfit(case$Z, case$y, linear = case$X, folds = 3, width = 6,
                       epochs = 120, seed = 1)

  expect_true(cf$has_linear)
  for (k in 1:3) expect_length(cf$bodies[[k]]$aux, 1)
  expect_true(any(grepl("ehat_treat", colnames(feature_matrix(cf)))))
  # One extra column per fold relative to no linear terms.
  plain <- bllnn_crossfit(case$Z, case$y, folds = 3, width = 6, epochs = 120,
                          seed = 1)
  expect_equal(cf$m_per_fold, plain$m_per_fold + 1L)
})

# --- interface --------------------------------------------------------------

test_that("a crossfit object refuses newdata", {
  case <- cf_case(n = 120)
  cf <- bllnn_crossfit(case$Z, case$y, folds = 2, width = 4, epochs = 60,
                       seed = 1)

  expect_error(feature_matrix(cf, case$Z), "belongs to no fold")
  expect_error(bllnn_sampler(cf, tau2 = 1, data = case$Z),
               "already carries its features")
  expect_no_error(bllnn_sampler(cf, tau2 = 1))
})

test_that("a body still requires newdata", {
  case <- cf_case(n = 100)
  body <- bllnn_warmup(case$Z, case$y, width = 5, epochs = 60, seed = 1)

  expect_error(feature_matrix(body), "`newdata` is required")
  expect_error(feature_matrix(42), "bllnn_body or a bllnn_crossfit")
})

test_that("invalid fold specifications are refused", {
  case <- cf_case(n = 60)

  expect_error(bllnn_crossfit(case$Z, case$y, folds = 1), "at least 2")
  expect_error(bllnn_crossfit(case$Z, case$y, folds = 1000), "only 60 rows")
  expect_error(bllnn_crossfit(case$Z, case$y, folds = rep(1, 60)),
               "at least 2 distinct")
  expect_error(bllnn_crossfit(case$Z, case$y, folds = rep(1:2, 5)),
               "has length 10")
  expect_error(bllnn_crossfit(case$Z, case$y[1:10]), "length 10")
  expect_error(bllnn_crossfit(case$Z, case$y, folds = 25),
               "at least 4 rows outside")
})

test_that("print reports the split", {
  case <- cf_case(n = 120)
  cf <- bllnn_crossfit(case$Z, case$y, folds = 3, width = 4, epochs = 60,
                       seed = 1)

  expect_output(print(cf), "bllnn_crossfit")
  expect_output(print(cf), "3 over 120")
  expect_output(print(cf), "block diagonal")
})

# --- the point of doing this at all ----------------------------------------

test_that("cross-fitting fits f at least as well as a single split", {
  skip_on_cran()
  case <- cf_case(n = 400)
  truth <- case$f_true - mean(case$f_true)

  # Both arms must converge, or the comparison measures which tolerates
  # truncation better rather than which fits f better -- at 800 epochs with
  # the default learning rate this assertion reverses for exactly that reason.
  # A brisker learning rate than the package default is used deliberately, so
  # the test isolates cross-fitting from the choice of default step size.
  n_epochs <- 2000
  lr <- 0.03

  cf <- bllnn_crossfit(case$Z, case$y, folds = 5, width = 25,
                       epochs = n_epochs, learn_rate = lr, seed = 1)
  Phi_cf <- feature_matrix(cf)

  # The honest comparison: a single 50/50 split, which is what we had before,
  # can only fit the half it did not train on.
  set.seed(1)
  half <- sample(nrow(case$Z), nrow(case$Z) / 2)
  body <- bllnn_warmup(case$Z[half, , drop = FALSE], case$y[half],
                       width = 25, epochs = n_epochs, learn_rate = lr,
                       seed = 1)

  # Guard the premise: if either arm hit the budget it was still improving,
  # and the comparison below is not measuring what it claims to.
  expect_lt(body$best_epoch, n_epochs)
  for (k in seq_len(cf$n_folds)) {
    expect_lt(cf$bodies[[k]]$best_epoch, n_epochs)
  }
  Phi_split <- feature_matrix(body, case$Z[-half, , drop = FALSE])

  err_cf <- sqrt(mean((lm.fit(Phi_cf, truth)$fitted.values - truth)^2))
  t_split <- truth[-half]
  err_split <- sqrt(mean((lm.fit(Phi_split, t_split)$fitted.values - t_split)^2))

  cat(sprintf("\n[crossfit] approx err: 5-fold %.4f over %d rows | split %.4f over %d rows\n",
              err_cf, nrow(Phi_cf), err_split, nrow(Phi_split)))

  expect_lt(err_cf, err_split)
})
