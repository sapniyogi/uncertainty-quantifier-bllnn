# Reference implementation, plain R by design.
#
# The network body: a single hidden layer trained once on a seed fold, then
# frozen. Deliberately dependency-free. A torch backend was considered and
# deferred: the optimiser is not the binding constraint here (Adam drives the
# training loss to ~1e-4 on this problem), the residual error is the
# approximation error of a single layer on finite data, and at this scale
# R's matrix operations are competitive with per-operation dispatch through
# libtorch. If a deeper body is ever wanted, the body interface below is the
# seam an alternative backend plugs into: anything supplying
# feature_matrix(body, newdata) works with the sampler unchanged.

activation_fun <- function(name) {
  switch(name,
         relu = function(a) pmax(a, 0),
         tanh = function(a) tanh(a))
}

activation_grad <- function(name) {
  switch(name,
         relu = function(a) (a > 0) * 1,
         tanh = function(a) 1 - tanh(a)^2)
}

#' Forward pass of the single-hidden-layer body
#'
#' @return A list with the pre-activations `A1`, the hidden activations `H`,
#'   and the scalar output `yhat`.
#' @noRd
mlp_forward <- function(par, X, act_f) {
  A1 <- X %*% par$W1 + rep(par$b1, each = nrow(X))
  H <- act_f(A1)
  list(A1 = A1, H = H, yhat = as.vector(H %*% par$w2 + par$b2))
}

#' Mean squared error and its exact gradient
#'
#' Separated from the optimiser so the analytic gradient can be checked
#' against finite differences, which is the closed-form test this routine
#' gets in place of a reference implementation.
#'
#' @noRd
mlp_loss_grad <- function(par, X, y, act_f, act_d) {
  n <- nrow(X)
  fw <- mlp_forward(par, X, act_f)
  resid <- fw$yhat - y

  g_yhat <- 2 * resid / n
  gA1 <- outer(g_yhat, par$w2) * act_d(fw$A1)

  list(
    loss = mean(resid^2),
    grad = list(
      W1 = crossprod(X, gA1),
      b1 = colSums(gA1),
      w2 = as.vector(crossprod(fw$H, g_yhat)),
      b2 = sum(g_yhat)
    )
  )
}

#' Initialise weights
#' @noRd
mlp_init <- function(p, width, activation) {
  scale_in <- if (activation == "relu") sqrt(2 / p) else sqrt(1 / p)
  list(
    W1 = matrix(stats::rnorm(p * width, sd = scale_in), p, width),
    b1 = rep(0, width),
    w2 = stats::rnorm(width, sd = sqrt(1 / width)),
    b2 = 0
  )
}

#' Learn a frozen network body
#'
#' Trains a single-hidden-layer network on a seed fold and freezes it. The
#' hidden activations become the fixed feature matrix that
#' [bllnn_sampler()] draws a last layer over.
#'
#' @details
#'
#' **Why the body is frozen.** Sampling conditions on the features. If they
#' were refit against the response being sampled, the conjugate draw would no
#' longer be exactly correct conditionally. Train once, here, on data the
#' sampler will not reuse; then never touch it again. For causal work the
#' calling code should cross-fit, training a body per fold and sampling on the
#' held-out one.
#'
#' **Training.** Full-batch Adam with decoupled weight decay on the weight
#' matrices but not the biases. A slice of the supplied data is held out for
#' validation; the parameters returned are those from the epoch with the best
#' validation loss, not the last epoch. That matters: without it the body
#' interpolates the training fold, and generalisation gets monotonically worse
#' with more epochs.
#'
#' **Standardisation.** Predictors are centred and scaled using statistics
#' computed here and stored on the object, so [feature_matrix()] applies the
#' identical transformation to new data. Recomputing them per-dataset would
#' silently produce a different feature map for the sampling fold.
#'
#' @section Confounding:
#'
#' When `linear` is supplied, an auxiliary body is trained for each of its
#' columns to predict that column from `x`, and the fitted values are appended
#' to the feature matrix. This gives the network the confounding channel
#' explicitly, and it is what keeps the linear coefficient honest.
#'
#' The alternative -- projecting the features onto the orthogonal complement of
#' the linear design -- is worse than doing nothing. It forces `X'Phi = 0`, so
#' the host's coefficient draw collapses to `(X'X)^-1 X'y`, which is ordinary
#' least squares with the confounding left in. Measured on 25 simulated
#' datasets at confounding 0.6, that scheme covered the true coefficient 8% of
#' the time against a nominal 95%; augmentation covered it consistently with
#' nominal. See `inst/validation/confounding_checks.R`.
#'
#' @param x Predictor matrix or data frame, `n` x `p`, numeric. The variables
#'   the nonlinear function is over.
#' @param y Response for the warm-up fit, length `n`. In partial-linear use
#'   this is the part of the outcome the body is meant to explain.
#' @param linear Optional matrix or vector of linear-term predictors, with
#'   `n` rows. Supplying it adds one auxiliary feature per column, estimating
#'   the conditional mean of that column given `x`. Required for causal use;
#'   omit it when the linear part is absent or known to be independent of `x`.
#' @param width Number of hidden units.
#' @param epochs Maximum training epochs.
#' @param activation `"relu"` or `"tanh"`.
#' @param learn_rate Adam step size.
#' @param weight_decay Decoupled weight decay. `0` disables it. Values around
#'   `0.01` help; large values such as `0.1` over-smooth and degrade the fit.
#' @param validation Fraction of `x` held out to select the stopping epoch.
#' @param patience Stop after this many epochs with no validation improvement.
#' @param seed Optional integer seed. The random number stream of the caller is
#'   restored on exit.
#'
#' @return A `bllnn_body` object. Pass it to [feature_matrix()] to obtain the
#'   frozen features, or straight to [bllnn_sampler()] with `data`.
#'
#' @examples
#' sim <- sim_partial_linear(n = 200, p_z = 5, seed = 1)
#' r <- sim$data$y - as.vector(sim$X %*% sim$beta)
#'
#' body <- bllnn_warmup(sim$Z, r, width = 20, epochs = 300, seed = 1)
#' body
#' dim(feature_matrix(body, sim$Z))
#'
#' @seealso [feature_matrix()], [bllnn_sampler()]
#' @export
bllnn_warmup <- function(x, y, linear = NULL, width = 50, epochs = 2000,
                         activation = c("relu", "tanh"),
                         learn_rate = 0.01, weight_decay = 0.01,
                         validation = 0.25, patience = 300, seed = NULL) {
  activation <- match.arg(activation)

  if (is.data.frame(x)) x <- as.matrix(x)
  if (!is.matrix(x) || !is.numeric(x)) {
    stop("`x` must be a numeric matrix or data frame.", call. = FALSE)
  }
  if (anyNA(x)) stop("`x` must not contain NA.", call. = FALSE)
  if (!is.numeric(y) || anyNA(y)) {
    stop("`y` must be a numeric vector with no NAs.", call. = FALSE)
  }
  if (length(y) != nrow(x)) {
    stop(sprintf("`y` has length %d but `x` has %d rows.", length(y), nrow(x)),
         call. = FALSE)
  }
  if (!is.null(linear)) {
    if (is.data.frame(linear)) linear <- as.matrix(linear)
    if (is.vector(linear) && is.numeric(linear)) {
      linear <- matrix(linear, ncol = 1, dimnames = list(NULL, "linear1"))
    }
    if (!is.matrix(linear) || !is.numeric(linear)) {
      stop("`linear` must be a numeric matrix, data frame or vector.",
           call. = FALSE)
    }
    if (anyNA(linear)) stop("`linear` must not contain NA.", call. = FALSE)
    if (nrow(linear) != nrow(x)) {
      stop(sprintf("`linear` has %d rows but `x` has %d.",
                   nrow(linear), nrow(x)), call. = FALSE)
    }
    if (is.null(colnames(linear))) {
      colnames(linear) <- paste0("linear", seq_len(ncol(linear)))
    }
  }
  for (nm in c("width", "epochs", "patience")) {
    v <- get(nm)
    if (!is.numeric(v) || length(v) != 1 || is.na(v) || v < 1 || v != round(v)) {
      stop(sprintf("`%s` must be a single positive integer.", nm), call. = FALSE)
    }
  }
  if (!is.numeric(learn_rate) || length(learn_rate) != 1 || is.na(learn_rate) ||
      learn_rate <= 0) {
    stop("`learn_rate` must be a single positive number.", call. = FALSE)
  }
  if (!is.numeric(weight_decay) || length(weight_decay) != 1 ||
      is.na(weight_decay) || weight_decay < 0) {
    stop("`weight_decay` must be a single non-negative number.", call. = FALSE)
  }
  if (!is.numeric(validation) || length(validation) != 1 || is.na(validation) ||
      validation <= 0 || validation >= 1) {
    stop("`validation` must be a single number strictly between 0 and 1.",
         call. = FALSE)
  }
  if (nrow(x) < 4) {
    stop("`x` needs at least 4 rows to hold out a validation slice.",
         call. = FALSE)
  }

  if (!is.null(seed)) {
    if (exists(".Random.seed", envir = globalenv())) {
      old_seed <- get(".Random.seed", envir = globalenv())
      on.exit(assign(".Random.seed", old_seed, envir = globalenv()), add = TRUE)
    }
    set.seed(seed)
  }

  width <- as.integer(width)
  n <- nrow(x)
  p <- ncol(x)

  # Standardisation is part of the frozen map: store it, never recompute it.
  centre <- colMeans(x)
  scale_ <- apply(x, 2, stats::sd)
  scale_[!is.finite(scale_) | scale_ <= 0] <- 1
  X <- scale(x, center = centre, scale = scale_)
  attributes(X) <- list(dim = dim(X))

  y_centre <- mean(y)
  y_c <- y - y_centre

  n_val <- max(1L, round(validation * n))
  if (n_val >= n) n_val <- n - 1L
  val_idx <- sample.int(n, n_val)
  X_tr <- X[-val_idx, , drop = FALSE]; y_tr <- y_c[-val_idx]
  X_va <- X[val_idx, , drop = FALSE];  y_va <- y_c[val_idx]

  act_f <- activation_fun(activation)
  act_d <- activation_grad(activation)

  par <- mlp_init(p, width, activation)
  m_state <- lapply(par, function(z) z * 0)
  v_state <- lapply(par, function(z) z * 0)
  beta1 <- 0.9; beta2 <- 0.999; eps <- 1e-8
  decayed <- c("W1", "w2")

  best_val <- Inf
  best_par <- par
  best_epoch <- 0L
  since_improved <- 0L
  trace <- numeric(0)

  for (t in seq_len(epochs)) {
    lg <- mlp_loss_grad(par, X_tr, y_tr, act_f, act_d)
    for (nm in names(par)) {
      g <- lg$grad[[nm]]
      if (nm %in% decayed && weight_decay > 0) g <- g + weight_decay * par[[nm]]
      m_state[[nm]] <- beta1 * m_state[[nm]] + (1 - beta1) * g
      v_state[[nm]] <- beta2 * v_state[[nm]] + (1 - beta2) * g^2
      par[[nm]] <- par[[nm]] - learn_rate *
        (m_state[[nm]] / (1 - beta1^t)) /
        (sqrt(v_state[[nm]] / (1 - beta2^t)) + eps)
    }

    val_loss <- mean((mlp_forward(par, X_va, act_f)$yhat - y_va)^2)
    trace <- c(trace, val_loss)
    if (val_loss < best_val - 1e-10) {
      best_val <- val_loss
      best_par <- par
      best_epoch <- t
      since_improved <- 0L
    } else {
      since_improved <- since_improved + 1L
      if (since_improved >= patience) break
    }
  }

  # Auxiliary bodies for the confounding channel: one per linear term,
  # each estimating E[linear_j | x] on this same seed fold. Trained with
  # linear = NULL so the recursion terminates.
  aux <- NULL
  if (!is.null(linear)) {
    aux <- lapply(seq_len(ncol(linear)), function(j) {
      bllnn_warmup(x, linear[, j], linear = NULL, width = width,
                   epochs = epochs, activation = activation,
                   learn_rate = learn_rate, weight_decay = weight_decay,
                   validation = validation, patience = patience,
                   seed = if (is.null(seed)) NULL else seed + j)
    })
    names(aux) <- colnames(linear)
  }

  structure(list(
    params = best_par,
    activation = activation,
    width = width,
    n_inputs = p,
    input_names = colnames(x),
    centre = centre,
    scale = scale_,
    y_centre = y_centre,
    val_loss = best_val,
    best_epoch = best_epoch,
    epochs_run = length(trace),
    val_trace = trace,
    aux = aux,
    linear_names = if (is.null(linear)) NULL else colnames(linear)
  ), class = "bllnn_body")
}

#' Frozen features from a warmed-up body
#'
#' Applies the stored standardisation and returns the hidden activations with a
#' leading intercept column. This is the matrix [bllnn_sampler()] draws a last
#' layer over.
#'
#' @param body A `bllnn_body` from [bllnn_warmup()].
#' @param newdata Predictor matrix or data frame with the same columns used in
#'   training.
#'
#' @return A numeric matrix, `nrow(newdata)` x `(width + 1)`. The first column
#'   is the intercept.
#'
#' @examples
#' sim <- sim_partial_linear(n = 120, p_z = 5, seed = 1)
#' r <- sim$data$y - as.vector(sim$X %*% sim$beta)
#' body <- bllnn_warmup(sim$Z, r, width = 10, epochs = 200, seed = 1)
#' Phi <- feature_matrix(body, sim$Z)
#' dim(Phi)
#'
#' @export
feature_matrix <- function(body, newdata) {
  if (!inherits(body, "bllnn_body")) {
    stop("`body` must be a bllnn_body, as returned by bllnn_warmup().",
         call. = FALSE)
  }
  if (is.data.frame(newdata)) newdata <- as.matrix(newdata)
  if (!is.matrix(newdata) || !is.numeric(newdata)) {
    stop("`newdata` must be a numeric matrix or data frame.", call. = FALSE)
  }
  if (ncol(newdata) != body$n_inputs) {
    stop(sprintf("`newdata` has %d columns but the body was trained on %d.",
                 ncol(newdata), body$n_inputs), call. = FALSE)
  }
  if (anyNA(newdata)) stop("`newdata` must not contain NA.", call. = FALSE)

  X <- scale(newdata, center = body$centre, scale = body$scale)
  attributes(X) <- list(dim = dim(X))
  act_f <- activation_fun(body$activation)
  H <- act_f(X %*% body$params$W1 + rep(body$params$b1, each = nrow(X)))

  Phi <- cbind(1, H)
  colnames(Phi) <- c("intercept", paste0("h", seq_len(body$width)))

  # The confounding channel, if this body was warmed up with linear terms.
  if (!is.null(body$aux)) {
    ehat <- vapply(body$aux, function(a) predict(a, newdata),
                   numeric(nrow(newdata)))
    ehat <- matrix(ehat, nrow = nrow(newdata))
    colnames(ehat) <- paste0("ehat_", body$linear_names)
    Phi <- cbind(Phi, ehat)
  }
  Phi
}

#' Point predictions from the body itself
#'
#' The body's own output layer, kept for diagnostics. It is not what the
#' sampler uses -- the sampler replaces this layer with a posterior draw.
#'
#' @param object A `bllnn_body`.
#' @param newdata Predictor matrix or data frame.
#' @param ... Ignored.
#'
#' @return A numeric vector of predictions on the scale of the training
#'   response.
#'
#' @examples
#' sim <- sim_partial_linear(n = 120, p_z = 5, seed = 1)
#' r <- sim$data$y - as.vector(sim$X %*% sim$beta)
#' body <- bllnn_warmup(sim$Z, r, width = 10, epochs = 200, seed = 1)
#' head(predict(body, sim$Z))
#'
#' @export
predict.bllnn_body <- function(object, newdata, ...) {
  # Columns 2..(width+1) only: feature_matrix() may append confounding-channel
  # features after the hidden units, and the body's own output layer has
  # weights for the hidden units alone.
  Phi <- feature_matrix(object, newdata)
  H <- Phi[, seq_len(object$width) + 1L, drop = FALSE]
  as.vector(H %*% object$params$w2 + object$params$b2) + object$y_centre
}

#' @param x A `bllnn_body` object.
#' @param ... Ignored.
#' @rdname bllnn_warmup
#' @export
print.bllnn_body <- function(x, ...) {
  cat("<bllnn_body>\n")
  cat(sprintf("  architecture : %d -> %d (%s) -> 1\n",
              x$n_inputs, x$width, x$activation))
  n_aux <- if (is.null(x$aux)) 0L else length(x$aux)
  cat(sprintf("  features     : %d, including intercept%s\n",
              x$width + 1L + n_aux,
              if (n_aux > 0) sprintf(" and %d confounding channel(s)", n_aux) else ""))
  if (n_aux > 0) {
    cat(sprintf("  linear terms : %s\n", paste(x$linear_names, collapse = ", ")))
  }
  cat(sprintf("  epochs run   : %d, best at %d\n", x$epochs_run, x$best_epoch))
  cat(sprintf("  validation   : %.5g\n", x$val_loss))
  cat("  status       : frozen\n")
  invisible(x)
}
