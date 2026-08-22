# Reference implementation, plain R by design: per "Build order" in CLAUDE.md
# the R layer is the specification that any later C++ port is tested against.
#
# The linear algebra here is deliberately naive -- a plain solve() and a plain
# chol(), recomputed on every call. That is correct but wasteful: the fast path
# described under "Precomputation contract" in CLAUDE.md factors the cross
# product once at sampler construction and reuses it. That optimisation belongs
# with the sampler object, not here.

#' Posterior moments of the last-layer weights
#'
#' The exact conditional posterior of `w` when the residual has likelihood
#' `N(Phi w, sigma^2 I)` and the prior is `w ~ N(0, tau2 I)`. Split out from
#' [conjugate_draw()] so the closed form can be tested directly rather than
#' only through the randomness of a draw.
#'
#' @inheritParams conjugate_draw
#'
#' @return A list with `mean` (length `m`), `cov` (`m` x `m`), and `precision`
#'   (`m` x `m`, the inverse of `cov`).
#'
#' @noRd
conjugate_moments <- function(Phi, r, sigma, tau2) {
  if (!is.matrix(Phi) || !is.numeric(Phi)) {
    stop("`Phi` must be a numeric matrix.", call. = FALSE)
  }
  if (anyNA(Phi)) {
    stop("`Phi` must not contain NA.", call. = FALSE)
  }
  if (!is.numeric(r) || anyNA(r)) {
    stop("`r` must be a numeric vector with no NAs.", call. = FALSE)
  }
  if (length(r) != nrow(Phi)) {
    stop(sprintf("`r` has length %d but `Phi` has %d rows.",
                 length(r), nrow(Phi)), call. = FALSE)
  }
  if (!is.numeric(sigma) || length(sigma) != 1 || is.na(sigma) || sigma <= 0) {
    stop("`sigma` must be a single positive number. It is the noise standard ",
         "deviation, not the variance.", call. = FALSE)
  }
  if (!is.numeric(tau2) || length(tau2) != 1 || is.na(tau2) || tau2 <= 0) {
    stop("`tau2` must be a single positive number. It is the prior variance, ",
         "not the prior standard deviation.", call. = FALSE)
  }

  m <- ncol(Phi)

  # Posterior precision: A = Phi'Phi / sigma^2 + I / tau2.
  # The I / tau2 term keeps A positive definite even when m > n, so this is
  # well defined where ordinary least squares is not.
  precision <- crossprod(Phi) / sigma^2 + diag(m) / tau2

  covariance <- solve(precision)
  # solve() returns a matrix that is symmetric only up to rounding; chol() is
  # strict about that, so symmetrise before it is used as a covariance.
  covariance <- (covariance + t(covariance)) / 2

  mu <- as.vector(covariance %*% (crossprod(Phi, r) / sigma^2))

  list(mean = mu, cov = covariance, precision = precision)
}

#' One exact draw of the last-layer weights
#'
#' Draws a single vector `w` from the exact conditional posterior of the last
#' layer of a Bayesian Last Layer network, given frozen features.
#'
#' @details
#'
#' For frozen features `Phi`, residual `r`, known noise scale `sigma`, and prior
#' `w ~ N(0, tau2 I)`, the model
#'
#' \deqn{r \mid w \sim N(\Phi w, \sigma^2 I)}
#'
#' has the exactly Gaussian conditional posterior
#'
#' \deqn{w \mid r, \sigma^2 \sim N(V \Phi' r / \sigma^2, V), \quad
#'       V = (\Phi' \Phi / \sigma^2 + I / \tau^2)^{-1}.}
#'
#' This is drawn in closed form. No optimisation, MCMC, or variational
#' approximation is involved, which is what makes the result usable as an exact
#' Gibbs transition kernel.
#'
#' Two conventions matter and are easy to get backwards:
#' `sigma` is the noise **standard deviation**, while `tau2` is the prior
#' **variance**. Passing a variance as `sigma` will not error; it will quietly
#' return draws with the wrong spread.
#'
#' The noise scale is an argument, never estimated here. In a Gibbs sampler the
#' host owns `sigma^2` and updates it in its own step; estimating it inside this
#' draw would use the data twice and break the joint chain.
#'
#' Because the prior is proper, the draw is well defined even when `Phi` has
#' more columns than rows.
#'
#' @param Phi Fixed feature matrix, `n` x `m`. Frozen during sampling: the draw
#'   is exactly correct conditionally only because `Phi` does not depend on the
#'   response being fitted.
#' @param r Residual vector of length `n`, the part of the response this block
#'   is being asked to explain.
#' @param sigma Noise standard deviation, a single positive number. Supplied by
#'   the caller, never estimated internally.
#' @param tau2 Prior variance of each weight, a single positive number. The
#'   prior is `w ~ N(0, tau2 I)`.
#'
#' @return A numeric vector of length `m`: one draw of `w`. If `Phi` has column
#'   names they are carried through as the names of the result.
#'
#' @examples
#' set.seed(1)
#' Phi <- matrix(rnorm(100 * 4), nrow = 100, ncol = 4)
#' w_true <- c(1, -2, 0.5, 0)
#' r <- as.vector(Phi %*% w_true) + rnorm(100, sd = 0.5)
#'
#' conjugate_draw(Phi, r, sigma = 0.5, tau2 = 10)
#'
#' # Repeated draws trace out the posterior
#' draws <- t(replicate(1000, conjugate_draw(Phi, r, sigma = 0.5, tau2 = 10)))
#' colMeans(draws)
#'
#' @export
conjugate_draw <- function(Phi, r, sigma, tau2) {
  post <- conjugate_moments(Phi, r, sigma, tau2)

  # V = R'R with R upper triangular, so t(R) %*% z has covariance
  # t(R) I R = R'R = V, as required.
  R <- chol(post$cov)
  z <- stats::rnorm(length(post$mean))

  w <- as.vector(post$mean + t(R) %*% z)
  names(w) <- colnames(Phi)
  w
}
