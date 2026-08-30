# Reference implementation, plain R by design.
#
# The network body: one or more hidden layers trained once on a seed fold and
# then frozen. Depth is set by passing a vector to `width`.
#
# Deliberately dependency-free. A torch backend was considered and deferred:
# the optimiser is not the binding constraint at this scale (Adam drives the
# training loss to ~1e-4 on this problem), and R's matrix operations are
# competitive with per-operation dispatch through libtorch for networks this
# small. That argument weakens with real depth or real scale, and the body
# interface is the seam an alternative backend plugs into: anything supplying
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

#' Number of hidden layers carried by a parameter list
#' @noRd
mlp_layers <- function(par) sum(grepl("^W[0-9]+$", names(par)))

#' Forward pass through an arbitrary number of hidden layers
#'
#' @return A list with the per-layer pre-activations `A`, the per-layer
#'   activations `H`, and the scalar output `yhat`. The features the sampler
#'   draws over are the last element of `H`.
#' @noRd
mlp_forward <- function(par, X, act_f) {
  n_layer <- mlp_layers(par)
  n <- nrow(X)
  A <- vector("list", n_layer)
  H <- vector("list", n_layer)

  input <- X
  for (l in seq_len(n_layer)) {
    A[[l]] <- input %*% par[[paste0("W", l)]] +
      rep(par[[paste0("b", l)]], each = n)
    H[[l]] <- act_f(A[[l]])
    input <- H[[l]]
  }

  list(A = A, H = H,
       yhat = as.vector(H[[n_layer]] %*% par$w_out + par$b_out))
}

#' Mean squared error and its exact gradient
#'
#' Separated from the optimiser so the analytic gradient can be checked
#' against finite differences, which is the closed-form test this routine
#' gets in place of a reference implementation.
#'
#' Backpropagation over layers `L` down to 1:
#'   `delta_L = (dL/dyhat) w_out' * act'(A_L)`
#'   `delta_{l-1} = (delta_l W_l') * act'(A_{l-1})`
#' with `dL/dW_l = input_l' delta_l` and `dL/db_l = colSums(delta_l)`, where
#' `input_l` is `X` for the first layer and `H_{l-1}` after that.
#'
#' @noRd
mlp_loss_grad <- function(par, X, y, act_f, act_d) {
  n <- nrow(X)
  n_layer <- mlp_layers(par)
  fw <- mlp_forward(par, X, act_f)
  resid <- fw$yhat - y

  g_yhat <- 2 * resid / n
  grad <- list()
  grad$w_out <- as.vector(crossprod(fw$H[[n_layer]], g_yhat))
  grad$b_out <- sum(g_yhat)

  delta <- outer(g_yhat, par$w_out) * act_d(fw$A[[n_layer]])
  for (l in rev(seq_len(n_layer))) {
    input_l <- if (l == 1L) X else fw$H[[l - 1L]]
    grad[[paste0("W", l)]] <- crossprod(input_l, delta)
    grad[[paste0("b", l)]] <- colSums(delta)
    if (l > 1L) {
      delta <- tcrossprod(delta, par[[paste0("W", l)]]) * act_d(fw$A[[l - 1L]])
    }
  }

  list(loss = mean(resid^2), grad = grad[names(par)])
}

#' Initialise weights for one or more hidden layers
#'
#' Each layer is scaled by its own fan-in, which matters more with depth: a
#' single global scale leaves deep networks either saturating or vanishing
#' before training starts.
#'
#' @noRd
mlp_init <- function(p, widths, activation) {
  gain <- if (activation == "relu") 2 else 1
  par <- list()
  fan_in <- p
  for (l in seq_along(widths)) {
    par[[paste0("W", l)]] <- matrix(
      stats::rnorm(fan_in * widths[l], sd = sqrt(gain / fan_in)),
      fan_in, widths[l])
    par[[paste0("b", l)]] <- rep(0, widths[l])
    fan_in <- widths[l]
  }
  par$w_out <- stats::rnorm(fan_in, sd = sqrt(1 / fan_in))
  par$b_out <- 0
  par
}

#' Train one network by full-batch Adam with decoupled weight decay
#'
#' Extracted from [bllnn_warmup()] so that several candidate settings can be
#' fitted against one fixed validation split. Returns the parameters from the
#' best-validation epoch, not the last.
#'
#' @noRd
mlp_train <- function(X_tr, y_tr, X_va, y_va, widths, activation,
                      learn_rate, weight_decay, epochs, patience) {
  act_f <- activation_fun(activation)
  act_d <- activation_grad(activation)

  par <- mlp_init(ncol(X_tr), widths, activation)
  m_state <- lapply(par, function(z) z * 0)
  v_state <- lapply(par, function(z) z * 0)
  beta1 <- 0.9; beta2 <- 0.999; eps <- 1e-8
  decayed <- c(paste0("W", seq_along(widths)), "w_out")

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

  list(par = best_par, val_loss = best_val, best_epoch = best_epoch,
       trace = trace)
}

#' Candidate settings searched when `tune = TRUE`
#'
#' Eight points spanning the region that a sweep over five simulated problems
#' found useful. Deliberately small: each candidate is a full training run.
#'
#' @noRd
tuning_grid <- function() {
  g <- expand.grid(activation = c("tanh", "relu"),
                   learn_rate = c(0.003, 0.03),
                   weight_decay = c(0.01, 0.1),
                   stringsAsFactors = FALSE)
  lapply(seq_len(nrow(g)), function(i) as.list(g[i, ]))
}

#' Learn a frozen network body
#'
#' Trains a network on a seed fold and freezes it. The activations of its last
#' hidden layer become the fixed feature matrix that [bllnn_sampler()] draws a
#' last layer over.
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
#' **Defaults, and why they are not the best-fitting ones.** A sweep over
#' eighteen settings and five simulated problems found `tanh` with
#' `learn_rate = 0.003` and `weight_decay = 0.1` fits `f` far better than these
#' defaults: on average 2.8 times closer, and up to 6 times on one problem.
#' Those settings were nevertheless rejected, because fitting `f` better made
#' the causal inference worse. Under them the 100-dataset coverage simulation
#' fell from 0.940 to 0.860 against a nominal 0.95, with all fourteen misses
#' entirely below the truth.
#'
#' The reason is that a sharper `f` absorbs more of the linear term, so the
#' bias in the coefficient grew from -0.049 to -0.070 while the intervals
#' narrowed. The intervals themselves stayed honest -- their width slightly
#' exceeded the spread of the estimator across datasets -- so this is bias
#' leaking through the confounding channel, not over-confidence. See
#' `inst/validation/coverage_simulation.R`.
#'
#' Treat that as a live limitation rather than a settled trade-off: the
#' defaults below pass the coverage gate, but they pass it partly because a
#' looser fit leaves less of the linear term to absorb. `tune = TRUE` searches
#' for the best validation loss, which is the right objective for prediction
#' and, on this evidence, the wrong one for causal work.
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
#' @param width Hidden layer sizes. A single number gives one hidden layer;
#'   a vector gives one layer per element, so `width = c(64, 32)` is a network
#'   with 64 units then 32. The features handed to the sampler are the
#'   activations of the last layer, so its size sets the number of features.
#' @param epochs Maximum training epochs.
#' @param activation `"tanh"` or `"relu"`.
#' @param learn_rate Adam step size.
#' @param weight_decay Decoupled weight decay, applied to the weight matrices
#'   and the output vector but never to the biases. `0` disables it.
#' @param validation Fraction of `x` held out to select the stopping epoch, and
#'   to choose among candidates when `tune = TRUE`.
#' @param patience Stop after this many epochs with no validation improvement.
#' @param tune Search a small grid of activations, learning rates and weight
#'   decays, keeping whichever minimises validation loss. Costs one training
#'   run per candidate, currently eight, and multiplies again across the
#'   auxiliary bodies when `linear` is supplied. Off by default for that
#'   reason, not because the defaults are always right.
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
                         validation = 0.25, patience = 300, tune = FALSE,
                         seed = NULL) {
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
  if (!is.numeric(width) || length(width) < 1 || anyNA(width) ||
      any(width < 1) || any(width != round(width))) {
    stop("`width` must be a positive integer, or a vector of them giving the ",
         "hidden layer sizes.", call. = FALSE)
  }
  for (nm in c("epochs", "patience")) {
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
  if (!is.logical(tune) || length(tune) != 1 || is.na(tune)) {
    stop("`tune` must be TRUE or FALSE.", call. = FALSE)
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

  widths <- as.integer(width)
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

  candidates <- if (isTRUE(tune)) tuning_grid() else
    list(list(activation = activation, learn_rate = learn_rate,
              weight_decay = weight_decay))

  # Every candidate is scored on the same validation split AND from the same
  # weight initialisation. Sharing the split alone is not enough: each fit
  # consumes random numbers to initialise, so without re-seeding, later
  # candidates would start from different weights and could win on a lucky
  # initialisation rather than on better settings.
  init_seed <- sample.int(.Machine$integer.max, 1L)
  fits <- lapply(candidates, function(cand) {
    set.seed(init_seed)
    mlp_train(X_tr, y_tr, X_va, y_va, widths, cand$activation,
              cand$learn_rate, cand$weight_decay, epochs, patience)
  })
  chosen <- which.min(vapply(fits, function(f) f$val_loss, numeric(1)))
  fit <- fits[[chosen]]
  activation <- candidates[[chosen]]$activation
  learn_rate <- candidates[[chosen]]$learn_rate
  weight_decay <- candidates[[chosen]]$weight_decay

  best_par <- fit$par
  best_val <- fit$val_loss
  best_epoch <- fit$best_epoch
  trace <- fit$trace

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
                   tune = tune,
                   seed = if (is.null(seed)) NULL else seed + j)
    })
    names(aux) <- colnames(linear)
  }

  structure(list(
    params = best_par,
    activation = activation,
    width = widths[length(widths)],
    widths = widths,
    n_inputs = p,
    input_names = colnames(x),
    centre = centre,
    scale = scale_,
    y_centre = y_centre,
    learn_rate = learn_rate,
    weight_decay = weight_decay,
    tuned = isTRUE(tune),
    tuning = if (isTRUE(tune)) {
      data.frame(
        activation = vapply(candidates, `[[`, "", "activation"),
        learn_rate = vapply(candidates, `[[`, 0, "learn_rate"),
        weight_decay = vapply(candidates, `[[`, 0, "weight_decay"),
        val_loss = vapply(fits, function(f) f$val_loss, numeric(1)),
        chosen = seq_along(fits) == chosen,
        stringsAsFactors = FALSE)
    } else NULL,
    val_loss = best_val,
    best_epoch = best_epoch,
    epochs_run = length(trace),
    val_trace = trace,
    aux = aux,
    linear_names = if (is.null(linear)) NULL else colnames(linear)
  ), class = "bllnn_body")
}

#' @details
#' For a `bllnn_body` this applies the stored standardisation and returns the
#' hidden activations with a leading intercept column, plus one confounding
#' channel per linear term if the body was warmed up with any.
#'
#' @rdname feature_matrix
#' @export
feature_matrix.bllnn_body <- function(object, newdata = NULL, ...) {
  body <- object
  if (is.null(newdata)) {
    stop("`newdata` is required: a bllnn_body stores the map, not the data ",
         "it was trained on.", call. = FALSE)
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

  # The features are the activations of the LAST hidden layer, which is what
  # makes this a Bayesian *last* layer. With one hidden layer that is the only
  # layer; with more, the earlier ones are representation and only the final
  # one is drawn over.
  fw <- mlp_forward(body$params, X, act_f)
  H <- fw$H[[length(fw$H)]]

  Phi <- cbind(1, H)
  colnames(Phi) <- c("intercept", paste0("h", seq_len(ncol(H))))

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
  as.vector(H %*% object$params$w_out + object$params$b_out) +
    object$y_centre
}

#' @param x A `bllnn_body` object.
#' @param ... Ignored.
#' @rdname bllnn_warmup
#' @export
print.bllnn_body <- function(x, ...) {
  cat("<bllnn_body>\n")
  cat(sprintf("  architecture : %s (%s)\n",
              paste(c(x$n_inputs, x$widths, 1), collapse = " -> "),
              x$activation))
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
