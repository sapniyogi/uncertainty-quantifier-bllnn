# Reference implementation, plain R by design: per "Build order" in CLAUDE.md
# the R layer is the specification that any later C++ port is tested against.

# --- outcome surfaces f(Z) -------------------------------------------------

# Friedman (1991), the standard BART/dbarts benchmark. Needs 5 columns of Z.
f_friedman <- function(Z) {
  10 * sin(pi * Z[, 1] * Z[, 2]) +
    20 * (Z[, 3] - 0.5)^2 +
    10 * Z[, 4] +
    5 * Z[, 5]
}

# Single-covariate smooth alternative. Needs 1 column of Z; any remaining
# columns are irrelevant noise, which is itself worth testing against.
f_sine <- function(Z) {
  2 * sin(2 * pi * Z[, 1])
}

# Nonlinear dependence of the k-th column of X on Z. Deliberately NOT equal to
# f(): if the confounding surface and the outcome surface were the same
# function, the bias would be far easier to remove than it is in practice.
m_confounder <- function(Z, k) {
  p_z <- ncol(Z)
  j1 <- ((k - 1) %% p_z) + 1
  j2 <- (k %% p_z) + 1
  sin(pi * Z[, j1]) + (Z[, j2] - 0.5)^2
}

#' Simulate data from a partial linear model
#'
#' Draws data from `y = X %*% beta + f(Z) + e`, the model this package targets,
#' and returns the realised truth alongside it so estimators can be scored
#' against a known answer.
#'
#' @section Why X is correlated with Z:
#'
#' The columns of `X` are generated as a nonlinear function of `Z` plus
#' independent noise, with the strength set by `confounding`. This is the point
#' of the simulator, not a complication of it. When `X` and `Z` are dependent, a
#' flexible fit of `f(Z)` can absorb part of `X %*% beta` and drag `beta` toward
#' zero -- the failure mode that design rule 1 (orthogonalise against the linear
#' terms) exists to prevent. Data with independent `X` and `Z` cannot exercise
#' that rule at all.
#'
#' At `confounding = 0` the columns of `X` are independent standard normals and
#' ordinary least squares of `y` on `X` is unbiased for `beta`. As `confounding`
#' rises, naive OLS becomes visibly biased. Both regimes are useful: the first
#' is the null case, the second is the reason for the package.
#'
#' @section Identification:
#'
#' In `y = X %*% beta + f(Z) + e` a constant can move freely between `f` and an
#' intercept, so the two are not separately identified. `center_f = TRUE` pins
#' this down by centring the realised `f(Z)` to sample mean zero. `X` carries no
#' intercept column for the same reason.
#'
#' @param n Number of observations. A single positive integer.
#' @param beta True linear coefficients. A numeric vector; its length sets the
#'   number of columns of `X`. If named, the names become the column names.
#' @param p_z Number of columns of `Z`, drawn independently as `Uniform(0, 1)`.
#'   Must be at least 5 for `f = "friedman"`, at least 1 otherwise.
#' @param f The nonlinear surface. Either `"friedman"` (Friedman 1991, the
#'   standard BART benchmark), `"sine"`, or a function taking the `Z` matrix and
#'   returning a numeric vector of length `n`.
#' @param confounding Strength of the dependence between `X` and `Z`, in
#'   `[0, 1)`. Roughly the correlation between each column of `X` and its
#'   nonlinear function of `Z`. At exactly 1 the columns of `X` would be
#'   deterministic functions of `Z` and `beta` would not be identified, so that
#'   value is refused.
#' @param sigma Standard deviation of the Gaussian noise. Must be non-negative;
#'   `sigma = 0` gives noiseless data, which is useful for testing exact
#'   recovery.
#' @param center_f Whether to centre the realised `f(Z)` to sample mean zero.
#'   See the Identification section.
#' @param seed Optional integer seed. If supplied, the random number stream of
#'   the caller is restored on exit, so calling this function does not disturb a
#'   surrounding simulation.
#'
#' @return A list with components:
#'   \describe{
#'     \item{`data`}{A data frame with columns `y`, the columns of `X`, and the
#'       columns of `Z`, suitable for a formula interface.}
#'     \item{`X`, `Z`}{The design matrices, for callers that want them directly.}
#'     \item{`beta`}{The true coefficient vector, named.}
#'     \item{`f_true`}{The realised `f(Z)`, after centring if requested.}
#'     \item{`mu_true`}{`X %*% beta + f_true`, the true conditional mean.}
#'     \item{`noise`}{The realised errors, so that `mu_true + noise` reproduces
#'       `y` exactly.}
#'     \item{`sigma`, `confounding`, `f_name`}{The settings used.}
#'   }
#'
#' @examples
#' sim <- sim_partial_linear(n = 200, beta = c(treat = 1.5), seed = 1)
#' str(sim$data)
#' sim$beta
#'
#' # Naive OLS is biased when X and Z are dependent
#' coef(lm(y ~ treat, data = sim$data))
#'
#' @export
sim_partial_linear <- function(n = 500,
                               beta = c(1, -0.5),
                               p_z = 5,
                               f = c("friedman", "sine"),
                               confounding = 0.6,
                               sigma = 1,
                               center_f = TRUE,
                               seed = NULL) {

  if (!is.numeric(n) || length(n) != 1 || is.na(n) || n < 1 || n != round(n)) {
    stop("`n` must be a single positive integer.", call. = FALSE)
  }
  if (!is.numeric(beta) || length(beta) < 1 || anyNA(beta)) {
    stop("`beta` must be a numeric vector of length >= 1 with no NAs.",
         call. = FALSE)
  }
  if (!is.numeric(p_z) || length(p_z) != 1 || is.na(p_z) ||
      p_z < 1 || p_z != round(p_z)) {
    stop("`p_z` must be a single positive integer.", call. = FALSE)
  }
  if (!is.numeric(confounding) || length(confounding) != 1 ||
      is.na(confounding) || confounding < 0 || confounding >= 1) {
    stop("`confounding` must be a single value in [0, 1). At 1 the columns of ",
         "X are deterministic functions of Z and `beta` is not identified; ",
         "use 0.99 to approach that limit.", call. = FALSE)
  }
  if (!is.numeric(sigma) || length(sigma) != 1 || is.na(sigma) || sigma < 0) {
    stop("`sigma` must be a single non-negative number.", call. = FALSE)
  }
  if (!is.logical(center_f) || length(center_f) != 1 || is.na(center_f)) {
    stop("`center_f` must be TRUE or FALSE.", call. = FALSE)
  }

  if (is.function(f)) {
    f_fun <- f
    f_name <- "custom"
    min_p_z <- 1L
  } else {
    f_name <- match.arg(f)
    f_fun <- switch(f_name, friedman = f_friedman, sine = f_sine)
    min_p_z <- switch(f_name, friedman = 5L, sine = 1L)
  }
  if (p_z < min_p_z) {
    stop(sprintf("f = \"%s\" needs at least %d columns in Z, but p_z = %d.",
                 f_name, min_p_z, p_z), call. = FALSE)
  }

  # Leave the random number stream of the caller exactly as we found it.
  if (!is.null(seed)) {
    if (exists(".Random.seed", envir = globalenv())) {
      old_seed <- get(".Random.seed", envir = globalenv())
      on.exit(assign(".Random.seed", old_seed, envir = globalenv()), add = TRUE)
    }
    set.seed(seed)
  }

  n <- as.integer(n)
  p_z <- as.integer(p_z)
  p_x <- length(beta)

  x_names <- names(beta)
  if (is.null(x_names) || anyNA(x_names) || any(!nzchar(x_names))) {
    x_names <- paste0("x", seq_len(p_x))
  }
  z_names <- paste0("z", seq_len(p_z))
  names(beta) <- x_names

  Z <- matrix(stats::runif(n * p_z), nrow = n, ncol = p_z,
              dimnames = list(NULL, z_names))

  # X = confounding * m(Z) + sqrt(1 - confounding^2) * noise, with m(Z)
  # standardised so that `confounding` reads as a correlation.
  M <- matrix(0, nrow = n, ncol = p_x)
  for (k in seq_len(p_x)) {
    mk <- m_confounder(Z, k)
    s <- if (n >= 2) stats::sd(mk) else 0
    M[, k] <- if (isTRUE(s > 0)) (mk - mean(mk)) / s else mk - mean(mk)
  }
  U <- matrix(stats::rnorm(n * p_x), nrow = n, ncol = p_x)
  X <- confounding * M + sqrt(1 - confounding^2) * U
  dimnames(X) <- list(NULL, x_names)

  f_true <- f_fun(Z)
  if (!is.numeric(f_true) || length(f_true) != n || anyNA(f_true)) {
    stop("`f` must return a numeric vector of length `n` with no NAs; got ",
         "length ", length(f_true), ".", call. = FALSE)
  }
  f_true <- as.vector(f_true)
  if (center_f) f_true <- f_true - mean(f_true)

  mu_true <- as.vector(X %*% beta) + f_true
  noise <- stats::rnorm(n, mean = 0, sd = sigma)
  y <- mu_true + noise

  list(
    data = data.frame(y = y, X, Z),
    X = X,
    Z = Z,
    beta = beta,
    f_true = f_true,
    mu_true = mu_true,
    noise = noise,
    sigma = sigma,
    confounding = confounding,
    f_name = f_name
  )
}
