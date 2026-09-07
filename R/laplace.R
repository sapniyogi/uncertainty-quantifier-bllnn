# Reference implementation, plain R by design.
#
# The Laplace approximation fits a Gaussian to the posterior by matching it at
# the mode: find the MAP weights, take the Hessian of the negative log
# posterior there, and use its inverse as the covariance. Unlike everything
# else in this package that produces a distribution, this one is approximate,
# and the approximation is not a detail to be tuned away -- it is the method.
#
# Two reasons it earns a place here anyway.
#
# For a Gaussian likelihood the log posterior is exactly quadratic, so the
# quadratic fit is not a fit at all: the Laplace moments equal the conjugate
# ones to machine precision. That makes this an internal version of the
# cross-check CLAUDE.md asks for against laplace-torch, runnable in the test
# suite instead of against a Python dependency.
#
# For counts it reaches somewhere the exact path cannot. Polya-Gamma covers the
# negative binomial and not the Poisson, because the augmentation identity
# needs the (1 + e^psi)^-b form that the Poisson likelihood does not have.
# Laplace has no such requirement. The two cover complementary ground rather
# than the same ground at different quality.
#
# What it is not is a Gibbs kernel, and kernel_table() keeps saying so.
# gibbs_step() refuses posterior = "laplace" without force = TRUE. A Gaussian
# fitted at the mode is not the conditional distribution of the weights, so
# iterating it does not leave the target invariant, and no amount of accuracy
# in the fit changes that.

#' Per-family pieces of the Newton step
#'
#' Everything the iteration needs that depends on the outcome model: the mean
#' given the linear predictor, the score, the observation weight, and the log
#' likelihood used to check that a step actually improved things.
#'
#' All three families use the canonical link, which is what makes the weight
#' the variance function alone. The observed Hessian and the expected Fisher
#' information then coincide, so Newton and Fisher scoring are the same
#' iteration, and the weighted cross-product is positive semi-definite by
#' construction -- with the ridge added it is positive definite, so `solve()`
#' never meets a saddle.
#'
#' @noRd
laplace_family <- function(family, sigma = NULL) {
  switch(
    family,
    gaussian = list(
      linkinv = function(eta) eta,
      score = function(y, mu) (y - mu) / sigma^2,
      weight = function(mu) rep(1 / sigma^2, length(mu)),
      loglik = function(y, eta) -sum((y - eta)^2) / (2 * sigma^2)
    ),
    binomial = list(
      linkinv = function(eta) stats::plogis(eta),
      score = function(y, mu) y - mu,
      weight = function(mu) mu * (1 - mu),
      # log(1 + e^eta) overflows for large eta and loses precision for very
      # negative eta; the pivot at zero is accurate on both sides.
      loglik = function(y, eta) {
        sum(y * eta - ifelse(eta > 0, eta + log1p(exp(-eta)), log1p(exp(eta))))
      }
    ),
    poisson = list(
      linkinv = function(eta) exp(eta),
      score = function(y, mu) y - mu,
      weight = function(mu) mu,
      loglik = function(y, eta) sum(y * eta - exp(eta))
    ),
    stop("Unsupported family.", call. = FALSE)
  )
}

laplace_families <- function() c("gaussian", "binomial", "poisson")

#' Gaussian approximation to the last-layer posterior
#'
#' Fits a Gaussian at the posterior mode: the MAP weights, and the inverse
#' Hessian of the negative log posterior there. For a Gaussian outcome this is
#' exact and agrees with the conjugate moments; for the others it is an
#' approximation, and the only one in this package.
#'
#' @details
#'
#' **The iteration.** The penalised log posterior is maximised by Newton steps
#'
#' \deqn{w \leftarrow w + (\Phi' W \Phi + I/\tau^2)^{-1}
#'       (\Phi' u - w/\tau^2),}
#'
#' with `u` the score and `W` the variance function on the diagonal. Each step
#' is halved as often as needed for the log posterior to increase, which is
#' what keeps the Poisson family from diverging when a large linear predictor
#' sends `exp(eta)` out of range. Convergence is on the size of the step.
#'
#' **When it is exact.** For `family = "gaussian"` the weight does not depend
#' on the linear predictor, so the log posterior is quadratic, the first step
#' from any starting point lands on the mode, and the Hessian is the same
#' everywhere. The returned moments then equal `conjugate_moments()` to machine
#' precision. This is asserted in the test suite, and it is the reason the two
#' code paths can be trusted against each other.
#'
#' **When it is not.** For binary and count outcomes the log posterior is not
#' quadratic. The mode is usually close to the posterior mean, but a Gaussian
#' fitted at the mode cannot represent skew, and the curvature at a single
#' point is a local statement about a global object. Expect the location to be
#' reasonable and the spread to be optimistic, more so as the information per
#' weight falls. For binary outcomes prefer `posterior = "polyagamma"`, which
#' is exact; this path exists for the Poisson case, which the augmentation
#' cannot reach, and for post-hoc uncertainty on a body that was trained
#' without any sampler at all.
#'
#' @param Phi Frozen feature matrix, `n` x `m`.
#' @param y Outcome of length `n`. On the natural scale of the family: a
#'   residual for `"gaussian"`, zeros and ones for `"binomial"`, non-negative
#'   integers for `"poisson"`.
#' @param tau2 Prior variance of each weight. The prior is `w ~ N(0, tau2 I)`.
#' @param family One of `"gaussian"`, `"binomial"`, `"poisson"`, each with its
#'   canonical link.
#' @param sigma Noise standard deviation, required for `"gaussian"` and refused
#'   for the others, which have no free scale parameter.
#' @param offset Known part of the linear predictor, length `n` or a single
#'   number. The host's contribution, as in [set_offset()].
#' @param max_iter Maximum Newton iterations.
#' @param tol Convergence tolerance on the largest absolute step.
#'
#' @return A list with `mean`, `cov`, `precision`, and `iterations`.
#'
#' @examples
#' set.seed(1)
#' Phi <- matrix(rnorm(200 * 3), 200, 3)
#' eta <- as.vector(Phi %*% c(1, -0.5, 0.25))
#'
#' # Binary outcome, no augmentation needed
#' y <- rbinom(200, 1, plogis(eta))
#' post <- laplace_moments(Phi, y, tau2 = 10, family = "binomial")
#' post$mean
#'
#' # For a Gaussian outcome it reproduces the exact conjugate moments, so it
#' # agrees with the closed form to machine precision rather than closely
#' r <- eta + rnorm(200, sd = 0.5)
#' lap <- laplace_moments(Phi, r, tau2 = 10, family = "gaussian", sigma = 0.5)
#' V <- solve(crossprod(Phi) / 0.5^2 + diag(3) / 10)
#' max(abs(lap$mean - as.vector(V %*% crossprod(Phi, r)) / 0.5^2))
#'
#' @seealso [laplace_draw()] for a draw from this approximation,
#'   [conjugate_draw()] for the exact Gaussian case.
#' @export
laplace_moments <- function(Phi, y, tau2, family = "gaussian", sigma = NULL,
                            offset = 0, max_iter = 100L, tol = 1e-10) {
  if (!is.matrix(Phi) || !is.numeric(Phi)) {
    stop("`Phi` must be a numeric matrix.", call. = FALSE)
  }
  if (anyNA(Phi)) {
    stop("`Phi` must not contain NA.", call. = FALSE)
  }
  if (!is.numeric(y) || anyNA(y)) {
    stop("`y` must be a numeric vector with no NAs.", call. = FALSE)
  }
  if (length(y) != nrow(Phi)) {
    stop(sprintf("`y` has length %d but `Phi` has %d rows.",
                 length(y), nrow(Phi)), call. = FALSE)
  }
  if (!is.numeric(tau2) || length(tau2) != 1 || is.na(tau2) || tau2 <= 0) {
    stop("`tau2` must be a single positive number. It is the prior variance, ",
         "not the prior standard deviation.", call. = FALSE)
  }
  if (!is.character(family) || length(family) != 1 ||
      !family %in% laplace_families()) {
    stop("`family` must be one of: ",
         paste(laplace_families(), collapse = ", "), ".", call. = FALSE)
  }

  if (family == "gaussian") {
    if (is.null(sigma)) {
      stop("`sigma` is required for family = \"gaussian\". The host owns the ",
           "noise scale and supplies it; it is never estimated here.",
           call. = FALSE)
    }
    if (!is.numeric(sigma) || length(sigma) != 1 || is.na(sigma) ||
        sigma <= 0) {
      stop("`sigma` must be a single positive number. It is the noise ",
           "standard deviation, not the variance.", call. = FALSE)
    }
  } else if (!is.null(sigma)) {
    stop(sprintf(paste("`sigma` applies only to family = \"gaussian\". The %s",
                       "likelihood has no free scale parameter."), family),
         call. = FALSE)
  }

  if (family == "binomial" && any(y != 0 & y != 1)) {
    stop("`y` must be 0 or 1 for family = \"binomial\".", call. = FALSE)
  }
  if (family == "poisson" && (any(y < 0) || any(y != round(y)))) {
    stop("`y` must be non-negative integers for family = \"poisson\".",
         call. = FALSE)
  }
  if (!is.numeric(offset) || anyNA(offset) ||
      (length(offset) != 1 && length(offset) != nrow(Phi))) {
    stop(sprintf("`offset` must be numeric with no NAs, of length 1 or %d.",
                 nrow(Phi)), call. = FALSE)
  }
  if (!is.numeric(max_iter) || length(max_iter) != 1 || is.na(max_iter) ||
      max_iter < 1) {
    stop("`max_iter` must be a single positive integer.", call. = FALSE)
  }

  fam <- laplace_family(family, sigma)
  n <- nrow(Phi)
  m <- ncol(Phi)
  offset <- rep_len(as.vector(offset), n)
  prior_precision <- diag(m) / tau2

  penalty <- function(w) -sum(w^2) / (2 * tau2)
  objective <- function(w, eta) fam$loglik(y, eta) + penalty(w)

  w <- rep(0, m)
  eta <- offset
  obj <- objective(w, eta)
  iterations <- 0L
  converged <- FALSE

  for (iter in seq_len(as.integer(max_iter))) {
    iterations <- iter
    mu <- fam$linkinv(eta)
    grad <- as.vector(crossprod(Phi, fam$score(y, mu)) - w / tau2)
    precision <- crossprod(Phi, Phi * fam$weight(mu)) + prior_precision
    step <- as.vector(solve(precision, grad))

    # Step-halving on the objective. The Newton direction is an ascent
    # direction because the precision is positive definite, so some small
    # enough step always improves; this only bounds how far we trust the
    # quadratic. Poisson is the family that needs it, where a long step can
    # push exp(eta) to Inf and the objective to NaN.
    #
    # The slack is defensive. A Newton step improves the objective by about
    # s'Hs/2, which converges quadratically to nothing while the objective
    # stays order n, so on the last iteration the comparison is made below one
    # ulp and in the noise. A strict >= survives that in practice -- an
    # increment far under an ulp rounds to zero and equality passes, and over
    # 100 random problems the strict rule never hit the halving floor -- but it
    # survives by luck, since eta is recomputed rather than accumulated and so
    # is not monotone to the last bit. A few ulps the wrong way would halve a
    # correct step into the floor and report a stall on a converged problem.
    slack <- 1e-10 * max(1, abs(obj))
    scale <- 1
    repeat {
      w_try <- w + scale * step
      eta_try <- offset + as.vector(Phi %*% w_try)
      obj_try <- objective(w_try, eta_try)
      if (is.finite(obj_try) && obj_try >= obj - slack) break
      scale <- scale / 2
      if (scale < 1e-10) {
        stop("The Newton iteration stalled: no step length improved the log ",
             "posterior. This usually means the features are badly scaled or ",
             "separate the outcome perfectly.", call. = FALSE)
      }
    }

    moved <- max(abs(w_try - w))
    w <- w_try
    eta <- eta_try
    obj <- obj_try
    if (moved < tol) {
      converged <- TRUE
      break
    }
  }

  if (!converged) {
    warning(sprintf(
      paste("The Newton iteration did not converge in %d steps; the mode is",
            "only approximate, and the covariance with it. Raise `max_iter`."),
      as.integer(max_iter)), call. = FALSE)
  }

  # The Hessian is evaluated at the mode, not at the last point a step was
  # taken from, so this recomputes rather than reusing the loop's copy.
  mu <- fam$linkinv(eta)
  precision <- crossprod(Phi, Phi * fam$weight(mu)) + prior_precision
  covariance <- solve(precision)
  covariance <- (covariance + t(covariance)) / 2

  names(w) <- colnames(Phi)
  list(mean = w, cov = covariance, precision = precision,
       iterations = iterations)
}

#' One draw from the Laplace approximation
#'
#' Draws from the Gaussian that [laplace_moments()] fits at the posterior mode.
#' Approximate by construction: see that function for what the approximation
#' does and does not preserve.
#'
#' @details
#' This is **not** a Gibbs transition kernel and must not be used as one. A
#' Gaussian fitted at the mode is not the conditional distribution of the
#' weights, so iterating it does not leave the target invariant.
#' [gibbs_step()] refuses `posterior = "laplace"` unless `force = TRUE`, and
#' that refusal is the intended behaviour rather than an obstacle.
#'
#' The legitimate uses are one-shot: uncertainty on a body trained outside any
#' sampler, and Poisson counts, which the exact augmentation cannot reach.
#'
#' **For more than a few draws, do not call this in a loop.** Every call
#' re-runs the whole Newton optimisation to find a mode that has not moved.
#' Call [laplace_moments()] once and sample from the `mean` and `cov` it
#' returns; the mode is the expensive part and it is the same for every draw.
#'
#' @inheritParams laplace_moments
#'
#' @return A numeric vector of length `m`: one draw of `w`, carrying the column
#'   names of `Phi` if it has any.
#'
#' @examples
#' set.seed(1)
#' Phi <- matrix(rnorm(200 * 3), 200, 3)
#' y <- rpois(200, exp(as.vector(Phi %*% c(0.5, -0.25, 0.1))))
#' laplace_draw(Phi, y, tau2 = 10, family = "poisson")
#'
#' @seealso [laplace_moments()], [conjugate_draw()]
#' @export
laplace_draw <- function(Phi, y, tau2, family = "gaussian", sigma = NULL,
                         offset = 0, max_iter = 100L, tol = 1e-10) {
  post <- laplace_moments(Phi, y, tau2, family = family, sigma = sigma,
                          offset = offset, max_iter = max_iter, tol = tol)
  draw_from_moments(post, colnames(Phi))
}
