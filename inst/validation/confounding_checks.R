# Why the network is given the confounding channel rather than blocked from it.
# Evidence for the identification strategy; referenced from ?bllnn_warmup.
#
#   Rscript inst/validation/confounding_checks.R      (several minutes)
#
# Three schemes for the same partial-linear problem:
#
#   none            features as learned, no allowance for confounding
#   orthogonalised  features projected onto the orthogonal complement of X,
#                   which is what design rule 1 originally prescribed
#   augmented       an estimate of E[X | Z] appended to the features
#
# The orthogonalised scheme is not a weaker fix, it is the wrong one. Forcing
# X'Phi = 0 makes the host's coefficient draw collapse to (X'X)^-1 X'y, so it
# reproduces ordinary least squares exactly, confounding and all. That is an
# algebraic identity, and the numbers below show it holding.

devtools::load_all(quiet = TRUE)

out_dir <- file.path("inst", "validation")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

BETA_TRUE <- 1.5
CONFOUNDING <- 0.6
SEEDS <- 101:125

source(file.path("inst", "validation", "host_sampler.R"))

one_dataset <- function(seed, n = 800, width = 40) {
  sim <- sim_partial_linear(n = n, beta = c(treat = BETA_TRUE), p_z = 5,
                            f = "friedman", confounding = CONFOUNDING,
                            sigma = 1, seed = seed)
  set.seed(seed + 5000)
  fold <- sample(rep(1:2, each = n / 2))
  A <- fold == 1; B <- fold == 2
  yB <- sim$data$y[B]; XB <- sim$X[B, , drop = FALSE]
  ZB <- sim$Z[B, , drop = FALSE]

  body <- bllnn_warmup(sim$Z[A, , drop = FALSE], sim$data$y[A],
                       linear = sim$X[A, , drop = FALSE],
                       width = width, epochs = 1200, seed = seed)
  Phi_aug <- feature_matrix(body, ZB)
  Phi_none <- Phi_aug[, !grepl("^ehat_", colnames(Phi_aug)), drop = FALSE]
  Phi_orth <- Phi_none - XB %*% qr.solve(crossprod(XB), crossprod(XB, Phi_none))

  schemes <- list(none = Phi_none, orthogonalised = Phi_orth,
                  augmented = Phi_aug)

  rows <- lapply(names(schemes), function(nm) {
    P <- schemes[[nm]]
    draws <- host_gibbs(yB, XB, P,
                        tau2 = var(yB) / mean(rowSums(P^2)),
                        seed = seed)$beta[, 1]
    ci <- quantile(draws, c(0.025, 0.975))
    data.frame(scheme = nm, seed = seed, mean = mean(draws),
               lo = ci[[1]], hi = ci[[2]],
               covers = BETA_TRUE >= ci[[1]] && BETA_TRUE <= ci[[2]],
               row.names = NULL)
  })
  res <- do.call(rbind, rows)
  res$ols <- unname(coef(lm(yB ~ XB))[2])
  res
}

cat(sprintf("beta = %.2f, confounding = %.1f, %d datasets, n = 800 (400 used)\n\n",
            BETA_TRUE, CONFOUNDING, length(SEEDS)))

res <- do.call(rbind, lapply(SEEDS, function(s) {
  cat("."); utils::flush.console()
  one_dataset(s)
}))
cat("\n\n")

summ <- do.call(rbind, lapply(split(res, res$scheme), function(d) {
  data.frame(scheme = d$scheme[1], datasets = nrow(d),
             mean_beta = mean(d$mean), bias = mean(d$mean) - BETA_TRUE,
             coverage = mean(d$covers), ci_width = mean(d$hi - d$lo))
}))
print(summ, digits = 4, row.names = FALSE)
cat(sprintf("\nnaive OLS across datasets: %.4f (bias %+.4f)\n",
            mean(res$ols[res$scheme == "none"]),
            mean(res$ols[res$scheme == "none"]) - BETA_TRUE))
cat("\nNote how closely the orthogonalised mean tracks naive OLS. They are the\n")
cat("same estimator: X'Phi = 0 removes the network from the coefficient draw.\n")

# --- figure ---------------------------------------------------------------

order_by <- res[res$scheme == "augmented", ]
ord <- order(order_by$mean)
cols <- c(none = "grey35", orthogonalised = "firebrick",
          augmented = "steelblue4")

png(file.path(out_dir, "confounding_intervals.png"),
    width = 1600, height = 560, res = 130)
par(mfrow = c(1, 3), mar = c(4.3, 4.3, 3.4, 1), oma = c(0, 0, 2.4, 0))
for (nm in c("none", "orthogonalised", "augmented")) {
  d <- res[res$scheme == nm, ][ord, ]
  plot(seq_len(nrow(d)), d$mean, ylim = range(res$lo, res$hi),
       pch = 16, cex = 0.7, col = ifelse(d$covers, cols[[nm]], "firebrick"),
       xlab = "dataset", ylab = expression(beta),
       main = sprintf("%s\ncoverage %.0f%%, bias %+.3f", nm,
                      100 * mean(d$covers), mean(d$mean) - BETA_TRUE))
  segments(seq_len(nrow(d)), d$lo, seq_len(nrow(d)), d$hi,
           col = ifelse(d$covers, cols[[nm]], "firebrick"))
  abline(h = BETA_TRUE, lwd = 1.8, lty = 2)
}
mtext(sprintf("dashed line is the true beta = %.2f; red intervals miss it",
              BETA_TRUE), outer = TRUE, line = 0.4, cex = 0.85)
dev.off()

cat("\nwrote confounding_intervals.png to", out_dir, "\n")
