# A host Gibbs sampler of the kind a statistician would write around this
# package, shared by the validation scripts so they exercise the same code.
#
#   y = X beta + f(Z) + eps
#
#   beta    | f, sigma^2  ~ conjugate normal
#   f       | beta, sigma^2 ~ gibbs_step() on our block
#   sigma^2 | beta, f      ~ inverse gamma
#
# Note what the block is and is not doing. It never sees y, only the residual
# y - X beta, and it never estimates sigma^2 -- the host owns that and passes
# it in each sweep. That is design rule 2, and this file is what it looks like
# from the caller's side.

host_gibbs <- function(y, X, Phi, tau2, n_iter = 1500, burn = 500,
                       v_beta = 100, a0 = 2, b0 = 1, seed = 1) {
  stopifnot(is.matrix(X), length(y) == nrow(X), nrow(Phi) == length(y))
  set.seed(seed)

  n <- length(y)
  p <- ncol(X)
  mod <- bllnn_sampler(Phi, tau2 = tau2)
  if (!is_valid_kernel(mod)) stop("sampler is not a valid kernel")

  XtX <- crossprod(X)
  beta <- rep(0, p)
  f <- rep(0, n)
  sigma2 <- var(y)

  keep_beta <- matrix(NA_real_, n_iter - burn, p)
  keep_sigma <- numeric(n_iter - burn)

  for (it in seq_len(n_iter)) {
    # beta | f, sigma^2
    Vb <- solve(XtX / sigma2 + diag(p) / v_beta)
    Vb <- (Vb + t(Vb)) / 2
    mb <- as.vector(Vb %*% (crossprod(X, y - f) / sigma2))
    beta <- as.vector(mb + t(chol(Vb)) %*% rnorm(p))

    # f | beta, sigma^2   -- one call into the package
    set_response(mod, y - as.vector(X %*% beta))
    set_sigma(mod, sqrt(sigma2))
    f <- gibbs_step(mod)

    # sigma^2 | beta, f
    resid <- y - as.vector(X %*% beta) - f
    sigma2 <- 1 / rgamma(1, a0 + n / 2, b0 + sum(resid^2) / 2)

    if (it > burn) {
      keep_beta[it - burn, ] <- beta
      keep_sigma[it - burn] <- sigma2
    }
  }
  list(beta = keep_beta, sigma2 = keep_sigma)
}

# Crude effective sample size from the autocorrelation function. Enough to
# tell a mixing chain from a stuck one, which is all the checkpoint needs.
effective_size <- function(x, max_lag = 200) {
  n <- length(x)
  a <- stats::acf(x, lag.max = min(max_lag, n - 2), plot = FALSE)$acf[-1]
  keep <- which(a < 0.05)[1]
  if (is.na(keep)) keep <- length(a)
  rho <- a[seq_len(max(1, keep - 1))]
  n / (1 + 2 * sum(rho))
}
