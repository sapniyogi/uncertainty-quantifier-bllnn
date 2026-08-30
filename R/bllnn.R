# The user-facing wrapper: a thin layer over the sampler object, never the
# other way round. Its job is to make the recipe that the coverage simulation
# validated -- cross-fit, confounding channel, partial out, tau2 from the data
# -- automatic, rather than four steps a user has to assemble correctly by
# hand. Each of those four was, at some point in development, the thing that
# was silently wrong.

#' Run the host Gibbs sampler around the block
#'
#' beta | f, sigma^2 conjugate normal; f | beta, sigma^2 from [gibbs_step()];
#' sigma^2 | beta, f inverse gamma. The block never sees the response, only the
#' residual, and never estimates sigma^2.
#'
#' @noRd
run_host_gibbs <- function(y, X, Phi, tau2, n_iter, burn, v_beta, a0, b0,
                           keep_f) {
  n <- length(y)
  p <- ncol(X)
  mod <- bllnn_sampler(Phi, tau2 = tau2)
  if (!is_valid_kernel(mod)) {
    stop("The sampler is not a valid Gibbs kernel: ", mod$reason, call. = FALSE)
  }

  XtX <- crossprod(X)
  prior_prec <- diag(p) / v_beta
  beta <- rep(0, p)
  f <- rep(0, n)
  sigma2 <- stats::var(y)

  n_keep <- n_iter - burn
  keep_beta <- matrix(NA_real_, n_keep, p, dimnames = list(NULL, colnames(X)))
  keep_sigma2 <- numeric(n_keep)
  keep_f_mat <- if (keep_f) matrix(NA_real_, n_keep, n) else NULL
  keep_w <- matrix(NA_real_, n_keep, ncol(Phi),
                   dimnames = list(NULL, colnames(Phi)))
  f_sum <- numeric(n)
  f_sumsq <- numeric(n)

  for (it in seq_len(n_iter)) {
    Vb <- solve(XtX / sigma2 + prior_prec)
    Vb <- (Vb + t(Vb)) / 2
    mb <- as.vector(Vb %*% (crossprod(X, y - f) / sigma2))
    beta <- as.vector(mb + t(chol(Vb)) %*% stats::rnorm(p))

    set_response(mod, y - as.vector(X %*% beta))
    set_sigma(mod, sqrt(sigma2))
    f <- gibbs_step(mod)

    resid <- y - as.vector(X %*% beta) - f
    sigma2 <- 1 / stats::rgamma(1, a0 + n / 2, b0 + sum(resid^2) / 2)

    if (it > burn) {
      i <- it - burn
      keep_beta[i, ] <- beta
      keep_sigma2[i] <- sigma2
      keep_w[i, ] <- mod$w
      f_sum <- f_sum + f
      f_sumsq <- f_sumsq + f^2
      if (keep_f) keep_f_mat[i, ] <- f
    }
  }

  list(beta = keep_beta, sigma2 = keep_sigma2, f = keep_f_mat,
       w = keep_w,
       f_mean = f_sum / n_keep,
       f_sd = sqrt(pmax(0, f_sumsq / n_keep - (f_sum / n_keep)^2)),
       weights = mod$w, sampler = mod)
}

#' Effective sample size from the autocorrelation function
#' @noRd
ess_of <- function(x, max_lag = 200) {
  n <- length(x)
  if (n < 10 || stats::sd(x) == 0) return(NA_real_)
  a <- stats::acf(x, lag.max = min(max_lag, n - 2), plot = FALSE)$acf[-1]
  cut <- which(a < 0.05)[1]
  if (is.na(cut)) cut <- length(a)
  n / (1 + 2 * sum(a[seq_len(max(1, cut - 1))]))
}

#' Fit a partial linear model with a Bayesian last layer
#'
#' The convenience interface: `y = X beta + f(Z) + e`, where `formula` gives
#' the response and the variables `f` is over, and `linear` gives the terms
#' that need honest credible intervals.
#'
#' @details
#'
#' This assembles the configuration that the package's coverage simulation
#' validates, in this order:
#'
#' 1. Cross-fitted network bodies, so every row's features come from a network
#'    that never saw that row's response ([bllnn_crossfit()]).
#' 2. The confounding channel: an auxiliary body per linear term estimating
#'    `E[X | Z]`, appended to the features.
#' 3. The linear terms residualised against it, via [partial_out()], so the
#'    network cannot absorb the coefficient.
#' 4. The prior variance read from the residual scale rather than guessed.
#'
#' Each of those steps was at some point in development the thing that was
#' silently wrong, and each is invisible in the output when it is. Getting all
#' four right unaided is not a reasonable expectation, which is the reason this
#' function exists.
#'
#' Around them it runs a Gibbs sampler: `beta` from its conjugate normal
#' conditional, `f` from [gibbs_step()], and `sigma^2` from its inverse gamma
#' conditional. The block itself never sees `y` and never estimates `sigma^2`.
#'
#' Over 100 simulated datasets at confounding 0.6, this configuration covered
#' the true coefficient 96% of the time against a nominal 95%, with bias
#' -0.020 where naive least squares carried +0.397. See
#' `inst/validation/coverage_simulation.R`.
#'
#' @param formula A formula `y ~ z1 + z2 + ...`. The right-hand side names the
#'   variables the nonlinear function is over. No intercept is fitted for them:
#'   the network carries its own.
#' @param data A data frame.
#' @param linear A one-sided formula naming the linear terms, for example
#'   `~ treat`. These get credible intervals. `NULL` fits `y = f(Z) + e` with
#'   no linear part.
#' @param folds Number of cross-fitting folds.
#' @param n_iter Total Gibbs iterations, including burn-in.
#' @param burn Iterations discarded as burn-in.
#' @param tau2 Prior variance of the last-layer weights; see
#'   [bllnn_sampler()]. `"auto"` reads it from the residual scale.
#' @param prior_beta Prior variance for the linear coefficients. The default is
#'   deliberately diffuse: these are the parameters of interest and should be
#'   driven by the data.
#' @param sigma_shape,sigma_rate Inverse gamma prior for `sigma^2`.
#' @param keep_f Store every draw of the fitted function. Needed for credible
#'   bands from [plot.bllnn_fit()]; costs `(n_iter - burn) * n` doubles.
#' @param seed Optional integer seed. The random number stream of the caller is
#'   restored on exit.
#' @param ... Passed to [bllnn_crossfit()] and on to [bllnn_warmup()]:
#'   `width`, `epochs`, `activation`, `learn_rate`, `weight_decay`, `tune`.
#'
#' @return An object of class `bllnn_fit`.
#'
#' @examples
#' sim <- sim_partial_linear(n = 150, beta = c(treat = 1.5), p_z = 5, seed = 1)
#' df <- sim$data
#'
#' fit <- bllnn(y ~ z1 + z2 + z3 + z4 + z5, data = df, linear = ~ treat,
#'              folds = 2, width = 6, epochs = 100,
#'              n_iter = 200, burn = 50, seed = 1)
#' fit
#' coef(fit)
#' confint(fit)
#'
#' @seealso [bllnn_crossfit()], [partial_out()], [bllnn_sampler()]
#' @export
bllnn <- function(formula, data, linear = NULL, folds = 5,
                  n_iter = 2000, burn = 500, tau2 = "auto",
                  prior_beta = 100, sigma_shape = 2, sigma_rate = 1,
                  keep_f = TRUE, seed = NULL, ...) {
  cl <- match.call()

  if (!inherits(formula, "formula") || length(formula) != 3L) {
    stop("`formula` must be two-sided, as in y ~ z1 + z2.", call. = FALSE)
  }
  if (missing(data) || !is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }
  for (nm in c("n_iter", "burn", "folds")) {
    v <- get(nm)
    if (!is.numeric(v) || length(v) != 1 || is.na(v) || v < 1 ||
        v != round(v)) {
      stop(sprintf("`%s` must be a single positive integer.", nm),
           call. = FALSE)
    }
  }
  if (burn >= n_iter) {
    stop(sprintf("`burn` is %d but `n_iter` is %d; nothing would be kept.",
                 burn, n_iter), call. = FALSE)
  }
  if (!is.logical(keep_f) || length(keep_f) != 1 || is.na(keep_f)) {
    stop("`keep_f` must be TRUE or FALSE.", call. = FALSE)
  }

  # One model frame for both parts, so a row dropped for a missing value is
  # dropped from all of them and y, Z and X cannot fall out of alignment.
  all_vars <- formula
  if (!is.null(linear)) {
    if (!inherits(linear, "formula") || length(linear) != 2L) {
      stop("`linear` must be a one-sided formula, as in ~ treat.",
           call. = FALSE)
    }
    all_vars <- stats::update(formula, paste(". ~ . +",
                                             deparse(linear[[2]])))
  }
  mf <- stats::model.frame(all_vars, data = data, na.action = stats::na.omit)
  if (nrow(mf) < 8) {
    stop(sprintf("Only %d complete rows after dropping missing values; ",
                 nrow(mf)),
         "cross-fitting needs more.", call. = FALSE)
  }

  y <- stats::model.response(mf)
  if (!is.numeric(y)) {
    stop("The response must be numeric. Binary and count outcomes need the ",
         "Polya-Gamma posterior, which is not implemented yet.", call. = FALSE)
  }

  Z <- stats::model.matrix(stats::delete.response(stats::terms(formula)), mf)
  Z <- Z[, colnames(Z) != "(Intercept)", drop = FALSE]
  if (ncol(Z) < 1) {
    stop("`formula` must name at least one variable for the nonlinear part.",
         call. = FALSE)
  }

  X <- NULL
  if (!is.null(linear)) {
    X <- stats::model.matrix(stats::terms(linear), mf)
    X <- X[, colnames(X) != "(Intercept)", drop = FALSE]
    if (ncol(X) < 1) {
      stop("`linear` must name at least one term.", call. = FALSE)
    }
  }

  if (!is.null(seed)) {
    if (exists(".Random.seed", envir = globalenv())) {
      old_seed <- get(".Random.seed", envir = globalenv())
      on.exit(assign(".Random.seed", old_seed, envir = globalenv()), add = TRUE)
    }
    set.seed(seed)
  }

  cf <- bllnn_crossfit(Z, y, linear = X, folds = folds,
                       seed = if (is.null(seed)) NULL else seed, ...)
  Phi <- feature_matrix(cf)

  # The linear design the host regresses on. Residualised, for the reason in
  # ?partial_out: Phi contains E[X|Z], so an un-residualised X would let the
  # network absorb part of the coefficient.
  X_used <- if (is.null(X)) {
    matrix(0, nrow = length(y), ncol = 0)
  } else {
    partial_out(cf)
  }

  if (ncol(X_used) == 0) {
    # No linear part: draw f and sigma^2 only.
    mod <- bllnn_sampler(Phi, tau2 = tau2)
    n_keep <- n_iter - burn
    keep_sigma2 <- numeric(n_keep)
    keep_f_mat <- if (keep_f) matrix(NA_real_, n_keep, length(y)) else NULL
    f <- rep(0, length(y)); sigma2 <- stats::var(y)
    f_sum <- numeric(length(y)); f_sumsq <- numeric(length(y))
    keep_w <- matrix(NA_real_, n_keep, ncol(Phi),
                     dimnames = list(NULL, colnames(Phi)))
    for (it in seq_len(n_iter)) {
      set_response(mod, y)
      set_sigma(mod, sqrt(sigma2))
      f <- gibbs_step(mod)
      resid <- y - f
      sigma2 <- 1 / stats::rgamma(1, sigma_shape + length(y) / 2,
                                  sigma_rate + sum(resid^2) / 2)
      if (it > burn) {
        i <- it - burn
        keep_sigma2[i] <- sigma2
        keep_w[i, ] <- mod$w
        f_sum <- f_sum + f; f_sumsq <- f_sumsq + f^2
        if (keep_f) keep_f_mat[i, ] <- f
      }
    }
    draws <- list(beta = matrix(NA_real_, n_keep, 0), sigma2 = keep_sigma2,
                  f = keep_f_mat, w = keep_w, f_mean = f_sum / n_keep,
                  f_sd = sqrt(pmax(0, f_sumsq / n_keep - (f_sum / n_keep)^2)),
                  sampler = mod)
  } else {
    draws <- run_host_gibbs(y, X_used, Phi, tau2 = tau2, n_iter = n_iter,
                            burn = burn, v_beta = prior_beta,
                            a0 = sigma_shape, b0 = sigma_rate,
                            keep_f = keep_f)
  }

  structure(list(
    call = cl,
    formula = formula,
    linear = linear,
    crossfit = cf,
    y = y,
    Z = Z,
    X = X,
    X_used = X_used,
    beta = draws$beta,
    sigma2 = draws$sigma2,
    f_draws = draws$f,
    f_weight_draws = draws$w,
    f_mean = draws$f_mean,
    f_sd = draws$f_sd,
    n_iter = n_iter,
    burn = burn,
    n = length(y),
    n_dropped = nrow(data) - nrow(mf),
    tau2 = draws$sampler$tau2,
    tau2_mode = draws$sampler$tau2_mode
  ), class = "bllnn_fit")
}
