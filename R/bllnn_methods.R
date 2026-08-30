# Methods on a fitted model. The question these answer is not "is it correct"
# but "would a statistician who has never met us read this and believe it".

#' @param x A `bllnn_fit`.
#' @param ... Ignored.
#' @rdname bllnn
#' @export
print.bllnn_fit <- function(x, ...) {
  cat("<bllnn_fit>\n")
  cat("Call: ", paste(deparse(x$call), collapse = " "), "\n\n", sep = "")
  cat(sprintf("  observations : %d%s\n", x$n,
              if (x$n_dropped > 0)
                sprintf("  (%d dropped for missing values)", x$n_dropped)
              else ""))
  cat(sprintf("  nonlinear in : %s\n",
              paste(colnames(x$Z), collapse = ", ")))
  cat(sprintf("  linear terms : %s\n",
              if (is.null(x$X)) "none" else paste(colnames(x$X), collapse = ", ")))
  cat(sprintf("  cross-fitting: %d folds, %d features per fold\n",
              x$crossfit$n_folds, x$crossfit$m_per_fold))
  cat(sprintf("  draws kept   : %d of %d\n", nrow(x$beta), x$n_iter))
  if (ncol(x$beta) > 0) {
    cat("\n")
    cm <- colMeans(x$beta)
    for (j in seq_along(cm)) {
      ci <- stats::quantile(x$beta[, j], c(0.025, 0.975))
      cat(sprintf("  %-12s %8.4f  [%.4f, %.4f]\n",
                  colnames(x$beta)[j], cm[j], ci[[1]], ci[[2]]))
    }
  }
  invisible(x)
}

#' Posterior means of the linear coefficients
#'
#' @param object A `bllnn_fit`.
#' @param ... Ignored.
#'
#' @return A named numeric vector.
#'
#' @examples
#' sim <- sim_partial_linear(n = 120, beta = c(treat = 1.5), p_z = 5, seed = 1)
#' fit <- bllnn(y ~ z1 + z2 + z3 + z4 + z5, data = sim$data, linear = ~ treat,
#'              folds = 2, width = 5, epochs = 80, n_iter = 150, burn = 50,
#'              seed = 1)
#' coef(fit)
#'
#' @export
coef.bllnn_fit <- function(object, ...) {
  if (ncol(object$beta) == 0) return(numeric(0))
  colMeans(object$beta)
}

#' Credible intervals for the linear coefficients
#'
#' Quantiles of the posterior draws. These are credible intervals, not
#' confidence intervals; the name follows the S3 convention so that existing
#' code works.
#'
#' @param object A `bllnn_fit`.
#' @param parm Coefficients to report. Defaults to all.
#' @param level Interval level.
#' @param ... Ignored.
#'
#' @return A matrix with lower and upper bounds.
#'
#' @examples
#' sim <- sim_partial_linear(n = 120, beta = c(treat = 1.5), p_z = 5, seed = 1)
#' fit <- bllnn(y ~ z1 + z2 + z3 + z4 + z5, data = sim$data, linear = ~ treat,
#'              folds = 2, width = 5, epochs = 80, n_iter = 150, burn = 50,
#'              seed = 1)
#' confint(fit)
#'
#' @export
confint.bllnn_fit <- function(object, parm, level = 0.95, ...) {
  if (!is.numeric(level) || length(level) != 1 || level <= 0 || level >= 1) {
    stop("`level` must be a single number strictly between 0 and 1.",
         call. = FALSE)
  }
  if (ncol(object$beta) == 0) {
    stop("This fit has no linear terms, so there are no coefficients to ",
         "report. Supply `linear` to bllnn().", call. = FALSE)
  }
  nms <- colnames(object$beta)
  if (missing(parm)) parm <- nms
  if (is.numeric(parm)) parm <- nms[parm]
  unknown <- setdiff(parm, nms)
  if (length(unknown)) {
    stop("Unknown coefficient(s): ", paste(unknown, collapse = ", "),
         ". Available: ", paste(nms, collapse = ", "), ".", call. = FALSE)
  }

  a <- (1 - level) / 2
  out <- t(apply(object$beta[, parm, drop = FALSE], 2, stats::quantile,
                 probs = c(a, 1 - a)))
  colnames(out) <- paste0(format(100 * c(a, 1 - a), trim = TRUE), " %")
  out
}

#' Summarise a fitted model
#'
#' @param object A `bllnn_fit`.
#' @param level Credible interval level.
#' @param ... Ignored.
#'
#' @return An object of class `summary.bllnn_fit`.
#'
#' @examples
#' sim <- sim_partial_linear(n = 120, beta = c(treat = 1.5), p_z = 5, seed = 1)
#' fit <- bllnn(y ~ z1 + z2 + z3 + z4 + z5, data = sim$data, linear = ~ treat,
#'              folds = 2, width = 5, epochs = 80, n_iter = 150, burn = 50,
#'              seed = 1)
#' summary(fit)
#'
#' @export
summary.bllnn_fit <- function(object, level = 0.95, ...) {
  a <- (1 - level) / 2
  tab <- NULL
  if (ncol(object$beta) > 0) {
    tab <- data.frame(
      mean = colMeans(object$beta),
      sd = apply(object$beta, 2, stats::sd),
      lower = apply(object$beta, 2, stats::quantile, probs = a),
      upper = apply(object$beta, 2, stats::quantile, probs = 1 - a),
      ess = apply(object$beta, 2, ess_of),
      row.names = colnames(object$beta)
    )
    # A coefficient whose interval excludes zero is the thing readers look
    # for first, so say it rather than making them compare two numbers.
    tab$excludes_zero <- tab$lower > 0 | tab$upper < 0
  }
  structure(list(
    call = object$call, table = tab, level = level,
    n = object$n, n_dropped = object$n_dropped,
    folds = object$crossfit$n_folds,
    m_per_fold = object$crossfit$m_per_fold,
    n_keep = length(object$sigma2), n_iter = object$n_iter,
    sigma = sqrt(mean(object$sigma2)),
    sigma_ess = ess_of(object$sigma2),
    tau2 = object$tau2, tau2_mode = object$tau2_mode,
    f_sd = stats::sd(object$f_mean)
  ), class = "summary.bllnn_fit")
}

#' @param x A `summary.bllnn_fit`.
#' @param ... Ignored.
#' @rdname summary.bllnn_fit
#' @export
print.summary.bllnn_fit <- function(x, ...) {
  cat("Partial linear model with a Bayesian last layer\n\n")
  cat("Call: ", paste(deparse(x$call), collapse = " "), "\n\n", sep = "")

  if (is.null(x$table)) {
    cat("No linear terms; only f(Z) and sigma were estimated.\n\n")
  } else {
    cat(sprintf("Linear coefficients (%.0f%% credible intervals):\n\n",
                100 * x$level))
    tab <- x$table
    out <- data.frame(
      Estimate = sprintf("%.4f", tab$mean),
      SD = sprintf("%.4f", tab$sd),
      Lower = sprintf("%.4f", tab$lower),
      Upper = sprintf("%.4f", tab$upper),
      ESS = sprintf("%.0f", tab$ess),
      ` ` = ifelse(tab$excludes_zero, "*", " "),
      check.names = FALSE, row.names = rownames(tab)
    )
    print(out)
    cat("\n* interval excludes zero\n")
  }

  cat(sprintf("\nresidual sd    : %.4f\n", x$sigma))
  cat(sprintf("sd of f(Z)     : %.4f\n", x$f_sd))
  cat(sprintf("observations   : %d%s\n", x$n,
              if (x$n_dropped > 0)
                sprintf(" (%d dropped for missing values)", x$n_dropped) else ""))
  cat(sprintf("cross-fitting  : %d folds, %d features per fold\n",
              x$folds, x$m_per_fold))
  cat(sprintf("draws          : %d kept of %d\n", x$n_keep, x$n_iter))
  cat(sprintf("prior variance : %.4g (%s)\n", x$tau2, x$tau2_mode))

  # Mixing is not a footnote. A coefficient whose chain has not mixed has an
  # interval that means nothing, whatever its width.
  worst <- if (is.null(x$table)) x$sigma_ess else min(c(x$table$ess, x$sigma_ess))
  cat(sprintf("smallest ESS   : %.0f of %d draws\n", worst, x$n_keep))
  if (worst < 100) {
    cat("\nWarning: effective sample size below 100. The chain has not mixed\n")
    cat("enough for these intervals to be trusted. Run longer, and look at\n")
    cat("plot(fit) before reporting anything.\n")
  }
  invisible(x)
}

#' Fitted values and predictions
#'
#' @details
#' With `newdata`, the nonlinear part is the average across the cross-fitting
#' folds of each body's features times that fold's weight draws. Cross-fitting
#' exists so that the conditional draw is exact for rows whose response is
#' being sampled; a new row has no response in the model, so there is nothing
#' to leak and every body is a legitimate estimate of `f`. Averaging them is an
#' ensemble, not a workaround.
#'
#' @param object A `bllnn_fit`.
#' @param newdata Optional data frame. Without it, fitted values for the rows
#'   the model was fitted to.
#' @param type `"response"` for `X beta + f(Z)`, or `"f"` for the nonlinear
#'   part alone.
#' @param interval Return posterior mean with a credible interval rather than
#'   the mean alone. Requires `keep_f = TRUE` at fitting time.
#' @param level Interval level.
#' @param ... Ignored.
#'
#' @return A numeric vector, or a matrix with `fit`, `lower` and `upper`.
#'
#' @examples
#' sim <- sim_partial_linear(n = 120, beta = c(treat = 1.5), p_z = 5, seed = 1)
#' fit <- bllnn(y ~ z1 + z2 + z3 + z4 + z5, data = sim$data, linear = ~ treat,
#'              folds = 2, width = 5, epochs = 80, n_iter = 150, burn = 50,
#'              seed = 1)
#' head(predict(fit))
#' head(predict(fit, type = "f", interval = TRUE))
#'
#' @export
predict.bllnn_fit <- function(object, newdata = NULL,
                              type = c("response", "f"),
                              interval = FALSE, level = 0.95, ...) {
  type <- match.arg(type)
  if (!is.logical(interval) || length(interval) != 1 || is.na(interval)) {
    stop("`interval` must be TRUE or FALSE.", call. = FALSE)
  }

  if (is.null(newdata)) {
    if (!interval) {
      out <- object$f_mean
      if (type == "response" && !is.null(object$X_used) &&
          ncol(object$X_used) > 0) {
        out <- out + as.vector(object$X_used %*% coef(object))
      }
      return(out)
    }
    if (is.null(object$f_draws)) {
      stop("Intervals need the stored draws. Refit with keep_f = TRUE.",
           call. = FALSE)
    }
    draws <- object$f_draws
    if (type == "response" && ncol(object$X_used) > 0) {
      draws <- draws + object$beta %*% t(object$X_used)
    }
  } else {
    if (!is.data.frame(newdata)) {
      stop("`newdata` must be a data frame.", call. = FALSE)
    }
    if (is.null(object$f_draws)) {
      stop("Predicting on new data needs the stored draws. Refit with ",
           "keep_f = TRUE.", call. = FALSE)
    }
    mf <- stats::model.frame(
      stats::delete.response(stats::terms(object$formula)), data = newdata,
      na.action = stats::na.pass)
    Znew <- stats::model.matrix(
      stats::delete.response(stats::terms(object$formula)), mf)
    Znew <- Znew[, colnames(Znew) != "(Intercept)", drop = FALSE]

    cf <- object$crossfit
    m_k <- cf$m_per_fold
    n_keep <- nrow(object$beta)
    if (n_keep == 0) n_keep <- length(object$sigma2)

    # Average each fold's body over the folds; see Details.
    acc <- matrix(0, nrow = n_keep, ncol = nrow(Znew))
    for (k in seq_len(cf$n_folds)) {
      Phi_k <- feature_matrix(cf$bodies[[k]], Znew)
      cols <- seq_len(m_k) + (k - 1) * m_k
      acc <- acc + object$f_weight_draws[, cols, drop = FALSE] %*% t(Phi_k)
    }
    draws <- acc / cf$n_folds

    if (type == "response") {
      if (is.null(object$linear)) {
        stop("`type = \"response\"` needs linear terms, and this fit has ",
             "none.", call. = FALSE)
      }
      stop("Predicting the response on new data needs E[X|Z] for those rows ",
           "to residualise them, which is not yet implemented. Use ",
           "type = \"f\".", call. = FALSE)
    }
  }

  if (!interval) return(colMeans(draws))
  a <- (1 - level) / 2
  cbind(fit = colMeans(draws),
        lower = apply(draws, 2, stats::quantile, probs = a),
        upper = apply(draws, 2, stats::quantile, probs = 1 - a))
}

#' Diagnostic plots for a fitted model
#'
#' Two panels per linear coefficient: a trace, which must look like noise, and
#' the posterior density. Then the fitted `f(Z)` with credible bands against
#' the response.
#'
#' A trace that drifts or wanders means the chain has not mixed, and the
#' intervals cannot be trusted whatever their width or the effective sample
#' size says.
#'
#' @param x A `bllnn_fit`.
#' @param which One or more of `"trace"`, `"density"`, `"fitted"`.
#' @param ... Passed to the underlying plot calls.
#'
#' @return `x`, invisibly. Called for the plots.
#'
#' @examples
#' sim <- sim_partial_linear(n = 120, beta = c(treat = 1.5), p_z = 5, seed = 1)
#' fit <- bllnn(y ~ z1 + z2 + z3 + z4 + z5, data = sim$data, linear = ~ treat,
#'              folds = 2, width = 5, epochs = 80, n_iter = 150, burn = 50,
#'              seed = 1)
#' plot(fit, which = "trace")
#'
#' @export
plot.bllnn_fit <- function(x, which = c("trace", "density", "fitted"), ...) {
  which <- match.arg(which, several.ok = TRUE)
  p <- ncol(x$beta)

  n_panel <- length(intersect(which, c("trace", "density"))) * max(p, 0) +
    ("fitted" %in% which)
  if (n_panel == 0) {
    stop("Nothing to plot: this fit has no linear terms and `which` did not ",
         "include \"fitted\".", call. = FALSE)
  }
  op <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(op), add = TRUE)
  graphics::par(mfrow = c(ceiling(n_panel / 2), min(n_panel, 2)),
                mar = c(4.2, 4.2, 3, 1))

  if (p > 0) {
    for (j in seq_len(p)) {
      nm <- colnames(x$beta)[j]
      if ("trace" %in% which) {
        plot(x$beta[, j], type = "l", col = "grey25", lwd = 0.6,
             xlab = "iteration after burn-in", ylab = nm,
             main = sprintf("trace: %s\nESS %.0f of %d", nm,
                            ess_of(x$beta[, j]), nrow(x$beta)), ...)
        graphics::abline(h = mean(x$beta[, j]), col = "steelblue4", lwd = 1.4)
      }
      if ("density" %in% which) {
        d <- stats::density(x$beta[, j])
        plot(d, main = sprintf("posterior: %s", nm), xlab = nm,
             col = "steelblue4", lwd = 1.6, ...)
        ci <- stats::quantile(x$beta[, j], c(0.025, 0.975))
        graphics::abline(v = ci, lty = 2, col = "firebrick")
        graphics::abline(v = 0, lty = 3, col = "grey40")
      }
    }
  }

  if ("fitted" %in% which) {
    ord <- order(x$f_mean)
    lo <- x$f_mean - 1.96 * x$f_sd
    hi <- x$f_mean + 1.96 * x$f_sd
    plot(seq_along(ord), x$f_mean[ord], type = "n",
         ylim = range(lo, hi),
         xlab = "observation, ordered by fitted f", ylab = "f(Z)",
         main = "fitted nonlinear part, 95% bands", ...)
    graphics::polygon(c(seq_along(ord), rev(seq_along(ord))),
                      c(lo[ord], rev(hi[ord])),
                      col = grDevices::rgb(0.2, 0.4, 0.8, 0.22), border = NA)
    graphics::lines(seq_along(ord), x$f_mean[ord], col = "steelblue4",
                    lwd = 1.6)
  }
  invisible(x)
}
