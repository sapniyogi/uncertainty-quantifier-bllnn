# Reference implementation, plain R by design.
#
# The object is an environment so that set_response() and set_sigma() mutate in
# place, matching dbarts' setResponse / setSigma. A host sampler calls these
# inside its own loop and must not have to reassign the object every sweep.

# Which posterior/feature combinations are legitimate Gibbs transition kernels.
# Only conjugate/frozen is usable today. The rest are listed so the guard
# refuses them by name and can say why, rather than failing obscurely later.
kernel_table <- function() {
  data.frame(
    posterior = c("conjugate", "conjugate", "polyagamma", "laplace",
                  "bootstrap", "vi", "sgmcmc"),
    features = c("frozen", "adaptive", "frozen", "frozen",
                 "frozen", "frozen", "frozen"),
    valid = c(TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE),
    reason = c(
      "Exact Gaussian conditional posterior on frozen features.",
      paste("Refitting the features on the response being sampled destroys",
            "the conditional exactness that frozen features buy."),
      "A valid kernel in principle, but not implemented yet.",
      "Approximate only. Not a conditional distribution, so not a kernel.",
      "Not a conditional distribution. Standalone mode only.",
      "A warm-up objective, not a draw.",
      "Correct only asymptotically. Research mode, with documented bias."
    ),
    stringsAsFactors = FALSE
  )
}

known_posteriors <- function() unique(kernel_table()$posterior)
known_features <- function() unique(kernel_table()$features)

stop_if_not_sampler <- function(mod) {
  if (!inherits(mod, "bllnn_sampler")) {
    stop("`mod` must be a bllnn_sampler object, as returned by ",
         "bllnn_sampler().", call. = FALSE)
  }
  invisible(TRUE)
}

#' Build a last-layer Gibbs sampler block
#'
#' Wraps a frozen feature matrix into a reusable object that supplies one step
#' of a Gibbs sampler written by the caller. The contract mirrors `dbarts`:
#' hand it a residual and a noise level, get back one posterior draw of the
#' nonlinear function.
#'
#' @details
#'
#' **Precomputation.** Frozen features means `Phi` is constant for the whole
#' run, so everything derivable from it alone is computed once here and stored:
#' the `m` x `m` cross-product `Phi'Phi` and the prior precision `I / tau2`.
#' [gibbs_step()] recomputes none of it. `Phi'r` is computed once per
#' [set_response()] call rather than once per step, so repeated draws against
#' the same residual cost only the solve.
#'
#' The linear algebra is deliberately naive -- a plain `solve()` per step. The
#' point of this structure is that the fast path, factoring the cross-product
#' once and reusing the basis, drops in later without any change to the API.
#'
#' **Reference semantics.** The returned object is an environment.
#' [set_response()] and [set_sigma()] modify it in place, so a host sampler
#' does not reassign it each sweep.
#'
#' **The host owns sigma.** The noise level is supplied through
#' [set_sigma()] and never estimated here. Estimating it internally would use
#' the data twice and break the joint chain.
#'
#' @param Phi Frozen feature matrix, `n` x `m`, or a `bllnn_body` from
#'   [bllnn_warmup()] together with `data`. Features are learned once on a
#'   held-out fold and then fixed; this object does not check that, but the
#'   exactness of the draw depends on it.
#' @param data Predictors to evaluate the features on. Required when `Phi` is a
#'   `bllnn_body`, and not permitted otherwise.
#' @param tau2 Prior variance of each weight, under `w ~ N(0, tau2 I)`. One of:
#'   a positive number, to fix it; `"auto"` (the default), to set it from the
#'   residual scale at the first [set_response()] call; or `"sample"`, to give
#'   it a conjugate inverse-gamma hyperprior and draw it in every
#'   [gibbs_step()]. Nobody should have to guess a prior variance, and the
#'   value matters: across its plausible range the linear coefficient moved by
#'   more than a third of its own size in our simulations.
#'
#'   `"auto"` and `"sample"` both calibrate to
#'   `var(r) / mean(rowSums(Phi^2))`, which is the scale on which `Phi w` has
#'   to live to explain the residual. That is empirical Bayes on the scale
#'   only; with `"sample"` the shape is fixed and the data move `tau2` from
#'   there.
#' @param tau2_shape Shape of the inverse-gamma hyperprior used by
#'   `tau2 = "sample"`. Must exceed 1 so the prior has a finite mean. The
#'   default of 2 gives a finite mean and infinite variance.
#' @param posterior Which posterior to draw from. Only `"conjugate"` is
#'   implemented; the other names in [valid_kernels()] are accepted so that
#'   [is_valid_kernel()] can report on them.
#' @param features Whether features stay fixed during sampling. Only
#'   `"frozen"` gives a valid kernel.
#'
#' @return A `bllnn_sampler` object.
#'
#' @examples
#' set.seed(1)
#' Phi <- matrix(rnorm(100 * 3), 100, 3)
#' r <- as.vector(Phi %*% c(1, -2, 0.5)) + rnorm(100, sd = 0.5)
#'
#' mod <- bllnn_sampler(Phi, tau2 = 10)
#' is_valid_kernel(mod)
#'
#' set_response(mod, r)
#' set_sigma(mod, 0.5)
#' f <- gibbs_step(mod)
#' length(f)
#'
#' @seealso [set_response()], [set_sigma()], [gibbs_step()],
#'   [is_valid_kernel()]
#' @export
bllnn_sampler <- function(Phi, tau2 = "auto", posterior = "conjugate",
                          features = "frozen", data = NULL,
                          tau2_shape = 2) {
  if (inherits(Phi, "bllnn_crossfit")) {
    if (!is.null(data)) {
      stop("A bllnn_crossfit already carries its features, so `data` is not ",
           "accepted.", call. = FALSE)
    }
    Phi <- feature_matrix(Phi)
  } else if (inherits(Phi, "bllnn_body")) {
    if (is.null(data)) {
      stop("When `Phi` is a bllnn_body, supply `data` so the frozen features ",
           "can be evaluated on it.", call. = FALSE)
    }
    Phi <- feature_matrix(Phi, data)
  } else if (!is.null(data)) {
    stop("`data` applies only when `Phi` is a bllnn_body. Pass the feature ",
         "matrix alone, or pass a body and its data.", call. = FALSE)
  }
  if (!is.matrix(Phi) || !is.numeric(Phi)) {
    stop("`Phi` must be a numeric matrix, or a bllnn_body with `data`.",
         call. = FALSE)
  }
  if (anyNA(Phi)) {
    stop("`Phi` must not contain NA.", call. = FALSE)
  }
  if (nrow(Phi) < 1 || ncol(Phi) < 1) {
    stop("`Phi` must have at least one row and one column.", call. = FALSE)
  }
  tau2_mode <- "fixed"
  if (is.character(tau2) && length(tau2) == 1 && tau2 %in% c("auto", "sample")) {
    tau2_mode <- tau2
    tau2 <- NA_real_
  } else if (!is.numeric(tau2) || length(tau2) != 1 || is.na(tau2) ||
             tau2 <= 0) {
    stop('`tau2` must be a single positive number, "auto", or "sample". It is ',
         "the prior variance, not the prior standard deviation.", call. = FALSE)
  }
  if (!is.numeric(tau2_shape) || length(tau2_shape) != 1 ||
      is.na(tau2_shape) || tau2_shape <= 1) {
    stop("`tau2_shape` must be a single number greater than 1, so the prior ",
         "has a finite mean.", call. = FALSE)
  }
  if (!is.character(posterior) || length(posterior) != 1 ||
      !posterior %in% known_posteriors()) {
    stop("`posterior` must be one of: ",
         paste(known_posteriors(), collapse = ", "), ".", call. = FALSE)
  }
  if (!is.character(features) || length(features) != 1 ||
      !features %in% known_features()) {
    stop("`features` must be one of: ",
         paste(known_features(), collapse = ", "), ".", call. = FALSE)
  }

  m <- ncol(Phi)
  tbl <- kernel_table()
  hit <- which(tbl$posterior == posterior & tbl$features == features)

  mod <- new.env(parent = emptyenv())

  # --- fixed for the lifetime of the object ---
  mod$Phi <- Phi
  mod$n <- nrow(Phi)
  mod$m <- m
  mod$tau2 <- tau2
  mod$tau2_mode <- tau2_mode
  mod$tau2_shape <- tau2_shape
  mod$tau2_rate <- NA_real_
  mod$posterior <- posterior
  mod$features <- features
  mod$valid <- if (length(hit) == 1) tbl$valid[hit] else FALSE
  mod$reason <- if (length(hit) == 1) {
    tbl$reason[hit]
  } else {
    sprintf("The combination posterior = \"%s\", features = \"%s\" is not a
             recognised kernel.", posterior, features)
  }

  # --- the precomputation contract: everything derivable from Phi alone ---
  mod$cross <- crossprod(Phi)
  mod$identity <- diag(m)
  mod$prior_precision <- if (tau2_mode == "fixed") mod$identity / tau2 else NULL

  # --- state supplied by the host, unset until it is ---
  mod$r <- NULL
  mod$Phi_r <- NULL
  mod$sigma <- NULL
  mod$w <- NULL
  mod$n_steps <- 0L

  class(mod) <- c("bllnn_sampler", "environment")
  mod
}

#' The table of valid posterior and feature combinations
#'
#' Not every posterior/feature pair is a legitimate Gibbs transition kernel.
#' This is the lookup [is_valid_kernel()] and [gibbs_step()] consult.
#'
#' @return A data frame with columns `posterior`, `features`, `valid` and
#'   `reason`.
#'
#' @examples
#' valid_kernels()
#'
#' @export
valid_kernels <- function() kernel_table()

#' Is this sampler a valid Gibbs transition kernel?
#'
#' Silently producing wrong inference is the worst failure mode available here,
#' so [gibbs_step()] refuses to run unless this is `TRUE`.
#'
#' @param mod A `bllnn_sampler` object.
#'
#' @return A single logical, with the explanation attached as the `reason`
#'   attribute.
#'
#' @examples
#' Phi <- matrix(rnorm(30), 10, 3)
#' is_valid_kernel(bllnn_sampler(Phi, tau2 = 1))
#' is_valid_kernel(bllnn_sampler(Phi, tau2 = 1, posterior = "bootstrap"))
#'
#' @export
is_valid_kernel <- function(mod) {
  stop_if_not_sampler(mod)
  out <- mod$valid
  attr(out, "reason") <- mod$reason
  out
}

#' Set the residual this block should explain
#'
#' Modifies `mod` in place; the caller does not reassign it. `Phi'r` is
#' computed here rather than in [gibbs_step()], so repeated draws against one
#' residual do not repeat the work.
#'
#' @param mod A `bllnn_sampler` object.
#' @param r Residual vector of length `n`.
#'
#' @return `mod`, invisibly.
#'
#' @examples
#' Phi <- matrix(rnorm(30), 10, 3)
#' mod <- bllnn_sampler(Phi, tau2 = 1)
#' set_response(mod, rnorm(10))
#'
#' @export
set_response <- function(mod, r) {
  stop_if_not_sampler(mod)
  if (!is.numeric(r) || anyNA(r)) {
    stop("`r` must be a numeric vector with no NAs.", call. = FALSE)
  }
  if (length(r) != mod$n) {
    stop(sprintf("`r` has length %d but this sampler was built on %d rows.",
                 length(r), mod$n), call. = FALSE)
  }
  mod$r <- as.vector(r)
  mod$Phi_r <- crossprod(mod$Phi, mod$r)

  # Calibrate the prior scale the first time a response arrives. The scale that
  # matters is the one on which f = Phi w lives, so match the implied prior
  # variance of f to the variance of the residual it has to explain.
  # Calibrate once, on the first response only. Recalibrating on every
  # set_response() would let the prior chase the residual it is meant to
  # regularise, and in a Gibbs loop that residual changes every sweep.
  if (mod$tau2_mode != "fixed" && is.na(mod$tau2)) {
    scale <- stats::var(mod$r) / mean(rowSums(mod$Phi^2))
    if (!is.finite(scale) || scale <= 0) scale <- 1
    if (mod$tau2_mode == "auto") {
      mod$tau2 <- scale
      mod$prior_precision <- mod$identity / scale
    } else {
      # tau2 ~ InvGamma(shape, rate) with the rate set so the prior mean is
      # that same scale. Empirical Bayes on the scale only; the shape is fixed
      # at a value giving a finite mean and infinite variance, so the data
      # move tau2 freely from there.
      mod$tau2_rate <- scale * (mod$tau2_shape - 1)
      mod$tau2 <- scale
      mod$prior_precision <- mod$identity / scale
    }
  }
  invisible(mod)
}

#' Set the noise level
#'
#' Modifies `mod` in place. `s` is the noise **standard deviation**, matching
#' `dbarts::setSigma()`. Passing a variance will not error; it will quietly
#' produce draws with the wrong spread.
#'
#' The host owns this quantity and updates it in its own Gibbs step. It is
#' never estimated inside this block.
#'
#' @param mod A `bllnn_sampler` object.
#' @param s Noise standard deviation, a single positive number.
#'
#' @return `mod`, invisibly.
#'
#' @examples
#' Phi <- matrix(rnorm(30), 10, 3)
#' mod <- bllnn_sampler(Phi, tau2 = 1)
#' set_sigma(mod, 0.5)
#'
#' @export
set_sigma <- function(mod, s) {
  stop_if_not_sampler(mod)
  if (!is.numeric(s) || length(s) != 1 || is.na(s) || s <= 0) {
    stop("`s` must be a single positive number. It is the noise standard ",
         "deviation, not the variance.", call. = FALSE)
  }
  mod$sigma <- s
  invisible(mod)
}

#' One posterior draw of the fitted function
#'
#' Draws the last-layer weights from their exact conditional posterior and
#' returns the fitted vector `Phi %*% w`. Each call consumes randomness, so
#' consecutive calls with identical inputs return different vectors -- that is
#' the point of a sampler rather than an estimator.
#'
#' @details
#'
#' Uses only the residual and the noise level, which are the only things that
#' change between iterations. The cross-product and prior precision were
#' computed at construction and are reused.
#'
#' The weight draw is stored on the object and is available as `mod$w`.
#'
#' @param mod A `bllnn_sampler` object with a response and sigma already set.
#' @param force Run even when [is_valid_kernel()] is `FALSE`. The resulting
#'   chain does not target the intended posterior. Only for deliberate
#'   experiments.
#'
#' @return A numeric vector of length `n`: one draw of the fitted function.
#'
#' @examples
#' set.seed(1)
#' Phi <- matrix(rnorm(100 * 3), 100, 3)
#' mod <- bllnn_sampler(Phi, tau2 = 10)
#' set_response(mod, as.vector(Phi %*% c(1, -2, 0.5)) + rnorm(100, sd = 0.5))
#' set_sigma(mod, 0.5)
#'
#' f1 <- gibbs_step(mod)
#' f2 <- gibbs_step(mod)
#' identical(f1, f2)   # FALSE: each call is a fresh draw
#' mod$w               # the weights behind the most recent draw
#'
#' @export
gibbs_step <- function(mod, force = FALSE) {
  stop_if_not_sampler(mod)
  if (!is.logical(force) || length(force) != 1 || is.na(force)) {
    stop("`force` must be TRUE or FALSE.", call. = FALSE)
  }

  if (!mod$valid && !force) {
    stop(sprintf(
      paste0("posterior = \"%s\" with features = \"%s\" is not a valid Gibbs ",
             "transition kernel, so gibbs_step() refuses to run.\n  %s\n  ",
             "Pass force = TRUE to override, accepting that the chain will ",
             "not target the intended posterior."),
      mod$posterior, mod$features, mod$reason), call. = FALSE)
  }

  if (is.null(mod$r)) {
    stop("No residual set. Call set_response(mod, r) before gibbs_step().",
         call. = FALSE)
  }
  if (is.null(mod$sigma)) {
    stop("No noise level set. Call set_sigma(mod, s) before gibbs_step().",
         call. = FALSE)
  }
  if (is.null(mod$prior_precision)) {
    stop("The prior scale is not calibrated yet. Call set_response(mod, r) ",
         "first; with tau2 = \"auto\" or \"sample\" the scale is read from ",
         "the residual.", call. = FALSE)
  }

  # tau2 | w, then w | r, sigma, tau2. Both steps are exact, so the pair is a
  # valid Gibbs sweep on (w, tau2) given the residual and the noise level.
  # tau2 is this block's own prior parameter, not the host's sigma^2, so
  # updating it here does not use the data twice.
  if (mod$tau2_mode == "sample" && !is.null(mod$w)) {
    shape <- mod$tau2_shape + mod$m / 2
    rate <- mod$tau2_rate + sum(mod$w^2) / 2
    mod$tau2 <- 1 / stats::rgamma(1, shape = shape, rate = rate)
    mod$prior_precision <- mod$identity / mod$tau2
  }

  post <- conjugate_moments_core(mod$cross, mod$Phi_r, mod$prior_precision,
                                 mod$sigma)
  w <- draw_from_moments(post, colnames(mod$Phi))

  mod$w <- w
  mod$n_steps <- mod$n_steps + 1L

  as.vector(mod$Phi %*% w)
}

#' @param x A `bllnn_sampler` object.
#' @param ... Ignored.
#'
#' @rdname bllnn_sampler
#' @export
print.bllnn_sampler <- function(x, ...) {
  cat("<bllnn_sampler>\n")
  cat(sprintf("  features   : %d x %d, %s\n", x$n, x$m, x$features))
  cat(sprintf("  posterior  : %s\n", x$posterior))
  cat(sprintf("  prior      : w ~ N(0, %s I)%s\n",
              if (is.na(x$tau2)) "tau2" else format(x$tau2, digits = 4),
              switch(x$tau2_mode,
                     fixed = "",
                     auto = "   [tau2 from the residual scale]",
                     sample = sprintf("   [tau2 sampled, InvGamma(%g, %s)]",
                                      x$tau2_shape,
                                      if (is.na(x$tau2_rate)) "rate"
                                      else format(x$tau2_rate, digits = 3)))))
  cat(sprintf("  valid kernel: %s\n", if (x$valid) "yes" else "NO"))
  if (!x$valid) cat(sprintf("    %s\n", x$reason))
  cat(sprintf("  response   : %s\n",
              if (is.null(x$r)) "not set" else "set"))
  cat(sprintf("  sigma      : %s\n",
              if (is.null(x$sigma)) "not set" else format(x$sigma)))
  cat(sprintf("  steps taken: %d\n", x$n_steps))
  invisible(x)
}
