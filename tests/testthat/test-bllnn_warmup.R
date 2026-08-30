warmup_case <- function(n = 200, p = 5) {
  sim <- sim_partial_linear(n = n, beta = c(treat = 1.5), p_z = p,
                            f = "friedman", confounding = 0.6, sigma = 1,
                            seed = 20260829)
  list(Z = sim$Z,
       r = sim$data$y - as.vector(sim$X %*% sim$beta),
       f_true = sim$f_true, sigma = sim$sigma, sim = sim)
}

# --- the closed-form check: analytic gradient vs finite differences ---------

test_that("the analytic gradient matches finite differences", {
  set.seed(4)
  n <- 30
  p <- 3
  width <- 6
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  for (act in c("relu", "tanh")) {
    act_f <- activation_fun(act)
    act_d <- activation_grad(act)

    set.seed(1)
    par <- mlp_init(p, width, act)
    # ReLU is not differentiable at 0, and a pre-activation landing exactly
    # there would make the comparison meaningless. Random continuous inputs
    # put that on a null set, but nudge the biases off zero to be sure.
    par$b1 <- par$b1 + rnorm(width, sd = 0.3)

    analytic <- mlp_loss_grad(par, X, y, act_f, act_d)$grad

    h <- 1e-6
    for (nm in names(par)) {
      numeric_grad <- par[[nm]] * 0
      for (i in seq_along(par[[nm]])) {
        up <- par; down <- par
        up[[nm]][i] <- up[[nm]][i] + h
        down[[nm]][i] <- down[[nm]][i] - h
        f_up <- mean((mlp_forward(up, X, act_f)$yhat - y)^2)
        f_down <- mean((mlp_forward(down, X, act_f)$yhat - y)^2)
        numeric_grad[i] <- (f_up - f_down) / (2 * h)
      }
      # Central differences at h = 1e-6 are accurate to roughly 1e-9 here;
      # 1e-6 leaves room for that without hiding a real error, which would
      # show up as an O(1) relative discrepancy.
      expect_equal(as.vector(analytic[[nm]]), as.vector(numeric_grad),
                   tolerance = 1e-6,
                   info = paste(act, nm))
    }
  }
})

test_that("the gradient check would fail on a wrong gradient", {
  # Guards the test above against passing vacuously.
  set.seed(4)
  X <- matrix(rnorm(30 * 3), 30, 3)
  y <- rnorm(30)
  act_f <- activation_fun("tanh")
  act_d <- activation_grad("tanh")
  set.seed(1)
  par <- mlp_init(3, 6, "tanh")

  correct <- mlp_loss_grad(par, X, y, act_f, act_d)$grad
  # A plausible slip: forgetting the activation derivative in the hidden layer.
  wrong_gA1 <- outer(2 * (mlp_forward(par, X, act_f)$yhat - y) / 30, par$w2)
  wrong_W1 <- crossprod(X, wrong_gA1)

  expect_false(isTRUE(all.equal(as.vector(correct$W1), as.vector(wrong_W1),
                                tolerance = 1e-6)))
})

# --- training behaviour -----------------------------------------------------

test_that("training reduces validation loss below the intercept-only fit", {
  case <- warmup_case()
  body <- bllnn_warmup(case$Z, case$r, width = 30, epochs = 800, seed = 1)

  # Predicting the mean is the trivial baseline; the body must beat it.
  expect_lt(body$val_loss, var(case$r))
  expect_gt(body$best_epoch, 0)
  expect_lte(body$best_epoch, body$epochs_run)
})

test_that("the returned parameters are the best-validation ones, not the last", {
  case <- warmup_case()
  body <- bllnn_warmup(case$Z, case$r, width = 40, epochs = 600,
                       patience = 1e6, seed = 1)

  # With patience effectively disabled the loop runs to the end, so if the
  # best epoch is earlier than the last, the returned parameters must come
  # from the checkpoint rather than from where training happened to stop.
  expect_equal(body$epochs_run, 600)
  expect_lt(body$best_epoch, 600)
  expect_equal(body$val_loss, min(body$val_trace))
})

test_that("early stopping halts before the epoch budget", {
  case <- warmup_case()
  body <- bllnn_warmup(case$Z, case$r, width = 30, epochs = 5000,
                       patience = 50, seed = 1)

  expect_lt(body$epochs_run, 5000)
  expect_lte(body$best_epoch + 50, body$epochs_run + 1)
})

test_that("trained features beat untrained random features", {
  case <- warmup_case(n = 400)
  set.seed(11)
  fold <- sample(rep(1:2, each = 200))
  ZA <- case$Z[fold == 1, , drop = FALSE]; rA <- case$r[fold == 1]
  ZB <- case$Z[fold == 2, , drop = FALSE]
  fB <- case$f_true[fold == 2] - mean(case$f_true[fold == 2])

  body <- bllnn_warmup(ZA, rA, width = 40, epochs = 1500, seed = 1)
  Phi_trained <- feature_matrix(body, ZB)

  # Matched-width random features, the honest baseline.
  set.seed(1)
  W <- matrix(rnorm(ncol(ZB) * 40), ncol(ZB), 40)
  Zs <- scale(ZB)
  attributes(Zs) <- list(dim = dim(Zs))
  Phi_random <- cbind(1, pmax(Zs %*% W, 0))

  err <- function(P) sqrt(mean((lm.fit(P, fB)$fitted.values - fB)^2))
  trained_err <- err(Phi_trained)
  random_err <- err(Phi_random)

  cat(sprintf("\n[warmup] approx err: trained %.4f vs random %.4f (sd(f)=%.3f)\n",
              trained_err, random_err, sd(fB)))

  expect_lt(trained_err, random_err)
})

# --- the frozen contract ----------------------------------------------------

test_that("features are deterministic once the body is frozen", {
  case <- warmup_case()
  body <- bllnn_warmup(case$Z, case$r, width = 20, epochs = 300, seed = 1)

  expect_identical(feature_matrix(body, case$Z), feature_matrix(body, case$Z))
})

test_that("standardisation is stored, not recomputed on new data", {
  case <- warmup_case(n = 300)
  train <- case$Z[1:150, , drop = FALSE]
  hold <- case$Z[151:300, , drop = FALSE]

  body <- bllnn_warmup(train, case$r[1:150], width = 15, epochs = 200, seed = 1)

  expect_equal(body$centre, colMeans(train))
  expect_equal(body$scale, apply(train, 2, sd))

  # The decisive check: evaluating the same rows inside a larger frame must
  # give identical features. If the scaling were recomputed per call it would
  # shift with the composition of newdata, and the sampler on a held-out fold
  # would silently see a different feature map from the one that was trained.
  both <- rbind(hold, train)
  expect_equal(feature_matrix(body, hold),
               feature_matrix(body, both)[seq_len(nrow(hold)), ])
})

test_that("feature_matrix shape and intercept column are as documented", {
  case <- warmup_case()
  body <- bllnn_warmup(case$Z, case$r, width = 12, epochs = 200, seed = 1)
  Phi <- feature_matrix(body, case$Z)

  expect_equal(dim(Phi), c(nrow(case$Z), 13L))
  expect_true(all(Phi[, 1] == 1))
  expect_equal(colnames(Phi)[1], "intercept")
  expect_equal(colnames(Phi)[2], "h1")
  # ReLU features are non-negative by construction.
  expect_true(all(Phi[, -1] >= 0))
})

test_that("a fixed seed reproduces the body, and the caller stream survives", {
  case <- warmup_case()

  a <- bllnn_warmup(case$Z, case$r, width = 15, epochs = 200, seed = 7)
  b <- bllnn_warmup(case$Z, case$r, width = 15, epochs = 200, seed = 7)
  d <- bllnn_warmup(case$Z, case$r, width = 15, epochs = 200, seed = 8)

  expect_equal(a$params, b$params)
  expect_false(isTRUE(all.equal(a$params$W1, d$params$W1)))

  set.seed(99); before <- runif(3)
  set.seed(99); invisible(bllnn_warmup(case$Z, case$r, width = 10,
                                       epochs = 100, seed = 1))
  after <- runif(3)
  expect_equal(before, after)
})

# --- integration with the sampler ------------------------------------------

test_that("a body can be handed straight to bllnn_sampler", {
  case <- warmup_case()
  body <- bllnn_warmup(case$Z, case$r, width = 20, epochs = 300, seed = 1)

  from_body <- bllnn_sampler(body, tau2 = 1, data = case$Z)
  from_matrix <- bllnn_sampler(feature_matrix(body, case$Z), tau2 = 1)

  expect_equal(from_body$cross, from_matrix$cross)
  expect_equal(from_body$m, body$width + 1L)
  expect_true(is_valid_kernel(from_body))

  set_response(from_body, case$r)
  set_sigma(from_body, case$sigma)
  f <- gibbs_step(from_body)
  expect_length(f, nrow(case$Z))
  expect_true(all(is.finite(f)))
})

test_that("the body and data arguments are validated together", {
  case <- warmup_case()
  body <- bllnn_warmup(case$Z, case$r, width = 10, epochs = 100, seed = 1)

  expect_error(bllnn_sampler(body, tau2 = 1), "supply `data`")
  expect_error(bllnn_sampler(feature_matrix(body, case$Z), tau2 = 1,
                             data = case$Z), "applies only when")
  expect_error(feature_matrix(body, case$Z[, 1:3]), "trained on 5")
  expect_error(feature_matrix(case$Z, case$Z), "must be a bllnn_body")
})

test_that("predict returns the body's own fit on the training scale", {
  case <- warmup_case()
  body <- bllnn_warmup(case$Z, case$r, width = 20, epochs = 400, seed = 1)
  p <- predict(body, case$Z)

  expect_length(p, nrow(case$Z))
  expect_true(all(is.finite(p)))
  # It should track the response it was trained on better than the mean does.
  expect_lt(mean((p - case$r)^2), var(case$r))
})

test_that("print reports the architecture and frozen status", {
  case <- warmup_case()
  body <- bllnn_warmup(case$Z, case$r, width = 11, epochs = 100, seed = 1)

  expect_output(print(body), "bllnn_body")
  expect_output(print(body), "5 -> 11")
  expect_output(print(body), "frozen")
})

# --- argument validation ----------------------------------------------------

test_that("invalid arguments are refused", {
  case <- warmup_case(n = 50)

  expect_error(bllnn_warmup(case$Z, case$r[1:10]), "length 10")
  expect_error(bllnn_warmup(case$Z, as.character(case$r)), "numeric")
  expect_error(bllnn_warmup(case$Z, case$r, width = 0), "positive integer")
  expect_error(bllnn_warmup(case$Z, case$r, width = 2.5), "positive integer")
  expect_error(bllnn_warmup(case$Z, case$r, epochs = 0), "positive integer")
  expect_error(bllnn_warmup(case$Z, case$r, learn_rate = 0), "positive")
  expect_error(bllnn_warmup(case$Z, case$r, weight_decay = -1), "non-negative")
  expect_error(bllnn_warmup(case$Z, case$r, validation = 0), "between 0 and 1")
  expect_error(bllnn_warmup(case$Z, case$r, validation = 1), "between 0 and 1")
  expect_error(bllnn_warmup(case$Z, case$r, activation = "sigmoid"), "arg")

  na_z <- case$Z; na_z[1, 1] <- NA
  expect_error(bllnn_warmup(na_z, case$r), "must not contain NA")
  expect_error(bllnn_warmup(matrix(rnorm(6), 3, 2), rnorm(3)), "at least 4 rows")
})

test_that("a constant predictor column does not produce NaN features", {
  # sd = 0 would divide by zero in the standardisation.
  set.seed(2)
  Z <- cbind(rnorm(60), rep(2.5, 60), rnorm(60))
  y <- Z[, 1] + rnorm(60, sd = 0.3)

  body <- bllnn_warmup(Z, y, width = 8, epochs = 150, seed = 1)
  Phi <- feature_matrix(body, Z)

  expect_equal(body$scale[2], 1)
  expect_true(all(is.finite(Phi)))
})

# --- the confounding channel -----------------------------------------------

test_that("supplying linear terms appends one feature per column", {
  case <- warmup_case(n = 200)
  X <- case$sim$X

  plain <- bllnn_warmup(case$Z, case$r, width = 12, epochs = 150, seed = 1)
  aug <- bllnn_warmup(case$Z, case$r, linear = X, width = 12, epochs = 150,
                      seed = 1)

  expect_null(plain$aux)
  expect_length(aug$aux, ncol(X))
  expect_equal(aug$linear_names, colnames(X))

  Phi_plain <- feature_matrix(plain, case$Z)
  Phi_aug <- feature_matrix(aug, case$Z)

  expect_equal(ncol(Phi_aug), ncol(Phi_plain) + ncol(X))
  expect_equal(tail(colnames(Phi_aug), 1), "ehat_treat")
  # The hidden block is untouched: augmentation appends, it does not alter.
  expect_equal(Phi_aug[, seq_len(ncol(Phi_plain))], Phi_plain)
})

test_that("the appended feature estimates the conditional mean of the linear term", {
  case <- warmup_case(n = 400)
  X <- case$sim$X

  body <- bllnn_warmup(case$Z, case$r, linear = X, width = 30, epochs = 800,
                       seed = 1)
  Phi <- feature_matrix(body, case$Z)
  ehat <- Phi[, "ehat_treat"]

  # It must carry real information about the treatment, or it cannot block
  # the confounding channel. Under confounding 0.6 the achievable correlation
  # is bounded well below 1, so this asserts a floor rather than a tight fit.
  expect_gt(abs(cor(ehat, X[, 1])), 0.3)
  expect_true(all(is.finite(ehat)))
})

test_that("augmentation does not disturb predict or the frozen contract", {
  case <- warmup_case(n = 200)
  body <- bllnn_warmup(case$Z, case$r, linear = case$sim$X, width = 12,
                       epochs = 200, seed = 1)

  p <- predict(body, case$Z)
  expect_length(p, nrow(case$Z))
  expect_true(all(is.finite(p)))

  expect_identical(feature_matrix(body, case$Z), feature_matrix(body, case$Z))
})

test_that("linear terms are validated", {
  case <- warmup_case(n = 60)

  expect_error(bllnn_warmup(case$Z, case$r, linear = matrix(0, 5, 1)),
               "has 5 rows")
  expect_error(bllnn_warmup(case$Z, case$r,
                            linear = matrix(NA_real_, nrow(case$Z), 1)),
               "must not contain NA")
  expect_error(bllnn_warmup(case$Z, case$r, linear = "treat"),
               "numeric matrix")
})

test_that("a bare vector of linear terms is accepted", {
  case <- warmup_case(n = 120)
  body <- bllnn_warmup(case$Z, case$r, linear = case$sim$X[, 1], width = 8,
                       epochs = 120, seed = 1)

  expect_length(body$aux, 1)
  expect_equal(colnames(feature_matrix(body, case$Z))[10], "ehat_linear1")
})

test_that("augmentation recovers the linear coefficient where plain features do not", {
  # End to end: a host Gibbs sampler around the block, which is the setting
  # the whole package exists to serve. Orthogonalising the features against X
  # is deliberately included because it is what design rule 1 originally
  # prescribed, and it fails badly enough to be worth pinning down.
  skip_on_cran()

  host_beta <- function(y, X, Phi, tau2, n_iter = 900, burn = 300, seed = 1) {
    set.seed(seed)
    n <- length(y); p <- ncol(X)
    mod <- bllnn_sampler(Phi, tau2 = tau2)
    XtX <- crossprod(X); beta <- rep(0, p); f <- rep(0, n); sigma2 <- var(y)
    keep <- matrix(NA_real_, n_iter - burn, p)
    for (it in seq_len(n_iter)) {
      Vb <- solve(XtX / sigma2 + diag(p) / 100)
      Vb <- (Vb + t(Vb)) / 2
      beta <- as.vector(Vb %*% (crossprod(X, y - f) / sigma2) +
                          t(chol(Vb)) %*% rnorm(p))
      set_response(mod, y - as.vector(X %*% beta))
      set_sigma(mod, sqrt(sigma2))
      f <- gibbs_step(mod)
      resid <- y - as.vector(X %*% beta) - f
      sigma2 <- 1 / rgamma(1, 2 + n / 2, 1 + sum(resid^2) / 2)
      if (it > burn) keep[it - burn, ] <- beta
    }
    keep[, 1]
  }

  beta_true <- 1.5
  sim <- sim_partial_linear(n = 600, beta = c(treat = beta_true), p_z = 5,
                            f = "friedman", confounding = 0.6, sigma = 1,
                            seed = 4242)
  set.seed(99)
  fold <- sample(rep(1:2, each = 300))
  A <- fold == 1; B <- fold == 2
  yB <- sim$data$y[B]; XB <- sim$X[B, , drop = FALSE]

  aug <- bllnn_warmup(sim$Z[A, , drop = FALSE], sim$data$y[A],
                      linear = sim$X[A, , drop = FALSE],
                      width = 30, epochs = 900, seed = 1)
  Phi_aug <- feature_matrix(aug, sim$Z[B, , drop = FALSE])
  Phi_plain <- Phi_aug[, !grepl("^ehat_", colnames(Phi_aug)), drop = FALSE]
  Phi_orth <- Phi_plain - XB %*% qr.solve(crossprod(XB),
                                          crossprod(XB, Phi_plain))

  tau_for <- function(P) var(yB) / mean(rowSums(P^2))
  b_aug <- host_beta(yB, XB, Phi_aug, tau_for(Phi_aug))
  b_orth <- host_beta(yB, XB, Phi_orth, tau_for(Phi_orth))
  ols <- unname(coef(lm(yB ~ XB))[2])

  cat(sprintf("\n[confounding] truth %.2f | augmented %.3f | orthogonalised %.3f | OLS %.3f\n",
              beta_true, mean(b_aug), mean(b_orth), ols))

  # Orthogonalising forces X'Phi = 0, so the coefficient draw reduces to OLS.
  # That is an algebraic identity, not a tendency, so it can be asserted tightly.
  expect_equal(mean(b_orth), ols, tolerance = 0.05)

  # Augmentation must land closer to the truth than that does.
  expect_lt(abs(mean(b_aug) - beta_true), abs(mean(b_orth) - beta_true))
  expect_lt(abs(mean(b_aug) - beta_true), 0.25)
})
