# Cross-fitting: K bodies, each trained without seeing the rows it will supply
# features for. Recovers the whole sample for inference instead of spending
# half of it on the seed fold.

#' Cross-fitted network bodies
#'
#' Trains one body per fold, each on the other folds, and assembles the frozen
#' feature matrix so that every row's features come from a body that never saw
#' that row's response. This is the default for causal use.
#'
#' @details
#'
#' **Why the features are block diagonal.** Each fold gets its own network, so
#' the `k`th body's hidden unit `h1` is a different function from the `j`th
#' body's `h1`. Sharing one weight vector across them would be incoherent.
#' Instead each fold's rows load only on that fold's columns, and the rest of
#' the row is zero. The posterior then factorises across blocks, so a draw from
#' the assembled system is exactly a set of independent per-fold draws — the
#' block structure is bookkeeping, not an approximation.
#'
#' The cost is `folds * (width + 1)` columns rather than `width + 1`. With a
#' naive solve that is `K^2` more work than solving the blocks separately;
#' exploiting the block structure is part of the same deferred fast path as the
#' eigendecomposition, and needs no API change.
#'
#' **What this object is for.** It carries the features for the rows it was
#' built on. It is not a predictor for new data: a new row belongs to no fold,
#' so there is no body that is out-of-sample for it. Prediction on fresh data
#' is a question for the user-facing wrapper, not for this object.
#'
#' @param x Predictor matrix or data frame, `n` x `p`.
#' @param y Response used to train the bodies, length `n`.
#' @param linear Optional linear-term matrix, as in [bllnn_warmup()]. Supplying
#'   it adds the confounding channel to every fold.
#' @param folds Number of folds, or an integer vector of length `n` assigning
#'   each row to a fold.
#' @param seed Optional integer seed. Controls both the fold split and the
#'   per-fold training; the random number stream of the caller is restored.
#' @param ... Passed to [bllnn_warmup()]: `width`, `epochs`, `activation`,
#'   `learn_rate`, `weight_decay`, `validation`, `patience`.
#'
#' @return A `bllnn_crossfit` object.
#'
#' @examples
#' sim <- sim_partial_linear(n = 150, p_z = 5, seed = 1)
#' cf <- bllnn_crossfit(sim$Z, sim$data$y, folds = 3, width = 8,
#'                      epochs = 150, seed = 1)
#' cf
#' dim(feature_matrix(cf))
#'
#' @seealso [bllnn_warmup()], [bllnn_sampler()]
#' @export
bllnn_crossfit <- function(x, y, linear = NULL, folds = 5, seed = NULL, ...) {
  if (is.data.frame(x)) x <- as.matrix(x)
  if (!is.matrix(x) || !is.numeric(x)) {
    stop("`x` must be a numeric matrix or data frame.", call. = FALSE)
  }
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
    if (!is.matrix(linear) || !is.numeric(linear) || nrow(linear) != nrow(x)) {
      stop("`linear` must be numeric with the same number of rows as `x`.",
           call. = FALSE)
    }
  }

  n <- nrow(x)

  if (!is.null(seed)) {
    if (exists(".Random.seed", envir = globalenv())) {
      old_seed <- get(".Random.seed", envir = globalenv())
      on.exit(assign(".Random.seed", old_seed, envir = globalenv()), add = TRUE)
    }
    set.seed(seed)
  }

  if (length(folds) == 1) {
    if (!is.numeric(folds) || is.na(folds) || folds < 2 || folds != round(folds)) {
      stop("`folds` must be a single integer of at least 2, or a fold vector.",
           call. = FALSE)
    }
    if (folds > n) {
      stop(sprintf("`folds` is %d but there are only %d rows.", folds, n),
           call. = FALSE)
    }
    fold <- sample(rep_len(seq_len(folds), n))
  } else {
    if (length(folds) != n) {
      stop(sprintf("`folds` has length %d but `x` has %d rows.",
                   length(folds), n), call. = FALSE)
    }
    fold <- as.integer(as.factor(folds))
  }
  k_folds <- max(fold)
  if (k_folds < 2) {
    stop("Cross-fitting needs at least 2 distinct folds.", call. = FALSE)
  }
  if (any(tabulate(fold, k_folds) < 4)) {
    stop("Every fold needs at least 4 rows outside it to train on; use fewer ",
         "folds or more data.", call. = FALSE)
  }

  bodies <- vector("list", k_folds)
  blocks <- vector("list", k_folds)

  for (k in seq_len(k_folds)) {
    train <- fold != k
    held <- !train
    bodies[[k]] <- bllnn_warmup(
      x[train, , drop = FALSE], y[train],
      linear = if (is.null(linear)) NULL else linear[train, , drop = FALSE],
      seed = if (is.null(seed)) NULL else seed + k, ...)
    blocks[[k]] <- feature_matrix(bodies[[k]], x[held, , drop = FALSE])
  }

  m_k <- ncol(blocks[[1]])
  block_names <- colnames(blocks[[1]])

  Phi <- matrix(0, nrow = n, ncol = k_folds * m_k)
  for (k in seq_len(k_folds)) {
    Phi[fold == k, seq_len(m_k) + (k - 1) * m_k] <- blocks[[k]]
  }
  colnames(Phi) <- as.vector(outer(block_names, seq_len(k_folds),
                                   function(nm, k) paste0("f", k, "_", nm)))

  structure(list(
    bodies = bodies,
    fold = fold,
    n_folds = k_folds,
    Phi = Phi,
    n = n,
    m_per_fold = m_k,
    has_linear = !is.null(linear)
  ), class = "bllnn_crossfit")
}

#' Frozen features
#'
#' @param object A `bllnn_body` from [bllnn_warmup()], or a `bllnn_crossfit`
#'   from [bllnn_crossfit()].
#' @param newdata Predictors to evaluate on. Required for a `bllnn_body`;
#'   must be `NULL` for a `bllnn_crossfit`, which carries the features for the
#'   rows it was built on.
#' @param ... Ignored.
#'
#' @return A numeric feature matrix.
#'
#' @examples
#' sim <- sim_partial_linear(n = 120, p_z = 5, seed = 1)
#' body <- bllnn_warmup(sim$Z, sim$data$y, width = 10, epochs = 150, seed = 1)
#' dim(feature_matrix(body, sim$Z))
#'
#' @export
feature_matrix <- function(object, newdata = NULL, ...) {
  UseMethod("feature_matrix")
}

#' @rdname feature_matrix
#' @export
feature_matrix.default <- function(object, newdata = NULL, ...) {
  stop("`object` must be a bllnn_body or a bllnn_crossfit.", call. = FALSE)
}

#' @rdname feature_matrix
#' @export
feature_matrix.bllnn_crossfit <- function(object, newdata = NULL, ...) {
  if (!is.null(newdata)) {
    stop("A bllnn_crossfit carries the features for the rows it was built on, ",
         "so `newdata` is not accepted. A new row belongs to no fold, and so ",
         "no body is out-of-sample for it.", call. = FALSE)
  }
  object$Phi
}

#' @param x A `bllnn_crossfit` object.
#' @param ... Ignored.
#' @rdname bllnn_crossfit
#' @export
print.bllnn_crossfit <- function(x, ...) {
  cat("<bllnn_crossfit>\n")
  cat(sprintf("  folds        : %d over %d rows\n", x$n_folds, x$n))
  cat(sprintf("  fold sizes   : %s\n",
              paste(tabulate(x$fold, x$n_folds), collapse = ", ")))
  cat(sprintf("  features     : %d per fold, %d total (block diagonal)\n",
              x$m_per_fold, ncol(x$Phi)))
  cat(sprintf("  confounding  : %s\n",
              if (x$has_linear) "linear terms supplied" else "none"))
  cat("  status       : frozen\n")
  invisible(x)
}
