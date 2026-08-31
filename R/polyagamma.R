# Exact sampling from the Polya-Gamma distribution.
#
# This is the piece that makes binary and count outcomes possible without an
# approximation. Conditional on a Polya-Gamma latent variable per observation,
# the logistic likelihood is exactly Gaussian in the weights, so the last-layer
# draw stays the closed-form conjugate one and the block stays a valid Gibbs
# kernel. Design rule 5 says exact draws over approximate ones, which rules out
# the truncated sum-of-gammas representation that is often used instead: it
# leaves a tail error of order 1/K after K terms, which is small but not zero.
#
# The method is Devroye's alternating-series sampler as given by Polson, Scott
# and Windle (2013). It proposes from a two-piece mixture -- an exponential
# right tail and a truncated inverse Gaussian left piece, split at t = 0.64 --
# then accepts or rejects using a series whose partial sums bracket the target
# density from alternating sides. Because the bounds alternate, the decision is
# exact after finitely many terms; nothing is truncated.

PG_TRUNC <- 0.64

#' Coefficients of the alternating series
#'
#' Two expressions for the same function, one convergent on each side of the
#' split point.
#'
#' @noRd
pg_coef <- function(n, x, t = PG_TRUNC) {
  if (x <= t) {
    pi * (n + 0.5) * (2 / (pi * x))^1.5 * exp(-2 * (n + 0.5)^2 / x)
  } else {
    pi * (n + 0.5) * exp(-(n + 0.5)^2 * pi^2 * x / 2)
  }
}

#' Inverse Gaussian distribution function with unit shape
#'
#' `mu = Inf` is the limit as `z` goes to zero and is reached in practice, so
#' it is handled rather than left to produce NaN.
#'
#' @noRd
pg_pigauss <- function(x, mu) {
  if (!is.finite(mu)) return(2 * stats::pnorm(-1 / sqrt(x)))
  z1 <- (x / mu - 1) / sqrt(x)
  z2 <- -(x / mu + 1) / sqrt(x)
  stats::pnorm(z1) + exp(2 / mu) * stats::pnorm(z2)
}

#' Inverse Gaussian truncated to (0, t]
#'
#' Two regimes. When the mean exceeds the truncation point, naive rejection
#' would almost always miss, so the draw comes from a truncated-normal
#' representation of `1/X` instead. Below it, plain rejection is efficient.
#'
#' @noRd
pg_rtigauss <- function(z, t = PG_TRUNC) {
  mu <- if (z > 1e-12) 1 / z else Inf

  if (mu > t) {
    repeat {
      repeat {
        e1 <- stats::rexp(1)
        e2 <- stats::rexp(1)
        if (e1^2 <= 2 * e2 / t) break
      }
      x <- t / (1 + t * e1)^2
      if (z <= 1e-12 || stats::runif(1) <= exp(-0.5 * z^2 * x)) return(x)
    }
  }

  repeat {
    y <- stats::rnorm(1)^2
    x <- mu + 0.5 * mu^2 * y - 0.5 * mu * sqrt(4 * mu * y + mu^2 * y^2)
    if (stats::runif(1) > mu / (mu + x)) x <- mu^2 / x
    if (x <= t) return(x)
  }
}

#' One exact draw from PG(1, z)
#' @noRd
rpg1_one <- function(z) {
  z <- abs(z) / 2
  t <- PG_TRUNC
  k <- pi^2 / 8 + z^2 / 2
  mu <- if (z > 1e-12) 1 / z else Inf

  p <- (pi / (2 * k)) * exp(-k * t)
  q <- 2 * exp(-z) * pg_pigauss(t, mu)
  mix <- p / (p + q)

  repeat {
    x <- if (stats::runif(1) < mix) {
      t + stats::rexp(1) / k
    } else {
      pg_rtigauss(z, t)
    }

    s <- pg_coef(0, x, t)
    y <- stats::runif(1) * s
    n <- 0L
    repeat {
      n <- n + 1L
      if (n %% 2L == 1L) {
        s <- s - pg_coef(n, x, t)
        if (y <= s) return(x / 4)
      } else {
        s <- s + pg_coef(n, x, t)
        if (y > s) break
      }
    }
  }
}

#' Draw from the Polya-Gamma distribution
#'
#' Exact draws from `PG(1, z)`, the augmentation that makes logistic
#' likelihoods conditionally Gaussian.
#'
#' @details
#'
#' **What it is for.** For a binary outcome the likelihood contribution is
#' `exp(psi)^y / (1 + exp(psi))`, which is not Gaussian in `psi` and leaves no
#' closed-form posterior for the weights. Polson, Scott and Windle (2013)
#' showed that introducing `omega ~ PG(1, psi)` per observation makes the
#' likelihood exactly Gaussian in `psi` given `omega`. Alternating between
#' drawing `omega` given the weights and the weights given `omega` is then an
#' exact Gibbs sampler, with no approximation at either step.
#'
#' **Why not the series representation.** `PG(1, z)` can be written as an
#' infinite weighted sum of exponentials, and truncating that sum is a common
#' shortcut. It is also an approximation, with a tail left over of order
#' `1 / K` after `K` terms. Design rule 5 asks for exact draws where they
#' exist, and here one does: Devroye's alternating-series method accepts or
#' rejects using partial sums that bracket the density from alternating sides,
#' so the decision is exact after finitely many terms.
#'
#' The mean is `tanh(z / 2) / (2 z)`, which tends to `1/4` as `z` tends to
#' zero.
#'
#' @param n Number of draws, or omit and pass a vector `z`.
#' @param z Tilting parameter. Recycled to length `n`; a vector gives one draw
#'   per element.
#'
#' @return A numeric vector of draws.
#'
#' @references
#' Polson, N. G., Scott, J. G. and Windle, J. (2013). Bayesian inference for
#' logistic models using Polya-Gamma latent variables. *Journal of the
#' American Statistical Association*, 108(504), 1339-1349.
#'
#' @examples
#' set.seed(1)
#' draws <- rpolyagamma(2000, z = 0)
#' c(mean = mean(draws), theoretical = 0.25)
#'
#' # The mean shrinks as the tilt grows
#' vapply(c(0, 1, 4), function(z) mean(rpolyagamma(500, z)), numeric(1))
#'
#' @export
rpolyagamma <- function(n, z = 0) {
  if (missing(n)) n <- length(z)
  if (!is.numeric(n) || length(n) != 1 || is.na(n) || n < 0 || n != round(n)) {
    stop("`n` must be a single non-negative integer.", call. = FALSE)
  }
  if (!is.numeric(z) || anyNA(z)) {
    stop("`z` must be numeric with no NAs.", call. = FALSE)
  }
  n <- as.integer(n)
  if (n == 0L) return(numeric(0))
  if (length(z) != n) {
    if (length(z) != 1 && n %% length(z) != 0) {
      warning("`n` is not a multiple of length(z); `z` will be recycled ",
              "partially.", call. = FALSE)
    }
    z <- rep_len(z, n)
  }
  vapply(z, rpg1_one, numeric(1))
}

#' Mean of PG(1, z)
#'
#' Closed form, used to check the sampler and available for diagnostics.
#'
#' @param z Tilting parameter.
#'
#' @return `tanh(z / 2) / (2 z)`, with the limit `1/4` taken at zero.
#'
#' @examples
#' pg_mean(c(0, 1, 4))
#'
#' @export
pg_mean <- function(z) {
  z <- abs(z)
  out <- ifelse(z < 1e-8, 0.25, tanh(z / 2) / (2 * pmax(z, 1e-300)))
  out
}

#' Variance of PG(1, z)
#'
#' @param z Tilting parameter.
#'
#' @return The variance, with the limit `1/24` taken at zero.
#'
#' @examples
#' pg_var(c(0, 1, 4))
#'
#' @export
pg_var <- function(z) {
  z <- abs(z)
  safe <- pmax(z, 1e-300)
  out <- (sinh(safe) - safe) / (4 * safe^3 * cosh(safe / 2)^2)
  ifelse(z < 1e-4, 1 / 24, out)
}
