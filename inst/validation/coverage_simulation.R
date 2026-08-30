# The money plot.
# Required by SESSION_PROTOCOL.md, "After the coverage simulation".
#
#   Rscript inst/validation/coverage_simulation.R          (roughly an hour)
#   Rscript inst/validation/coverage_simulation.R 10       (pilot, minutes)
#
# Across many simulated datasets, does a nominal 95% credible interval for the
# linear coefficient contain the truth about 95% of the time? Roughly five in a
# hundred should miss. This is the first test of the whole construction rather
# than any one part of it: simulator, cross-fitted bodies, confounding channel,
# conjugate draw and host sampler all have to be right together.
#
# Alongside it, a trace plot. It must look like noise. Visible drift or
# wandering means the chain is not mixing and the intervals cannot be trusted
# whatever the coverage number says.

devtools::load_all(quiet = TRUE)
source(file.path("inst", "validation", "host_sampler.R"))

out_dir <- file.path("inst", "validation")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

args <- commandArgs(trailingOnly = TRUE)
N_DATASETS <- if (length(args) >= 1) as.integer(args[1]) else 100

BETA_TRUE <- 1.5
CONFOUNDING <- 0.6
N_OBS <- 500
FOLDS <- 5
WIDTH <- 20
EPOCHS <- 600
N_ITER <- 1200
BURN <- 400

one_dataset <- function(seed, keep_draws = FALSE) {
  sim <- sim_partial_linear(n = N_OBS, beta = c(treat = BETA_TRUE), p_z = 5,
                            f = "friedman", confounding = CONFOUNDING,
                            sigma = 1, seed = seed)

  # Cross-fitted bodies with the confounding channel: the configuration the
  # package recommends for causal work.
  cf <- bllnn_crossfit(sim$Z, sim$data$y, linear = sim$X, folds = FOLDS,
                       width = WIDTH, epochs = EPOCHS, seed = seed)
  Phi <- feature_matrix(cf)
  tau2 <- var(sim$data$y) / mean(rowSums(Phi^2))

  fit <- host_gibbs(sim$data$y, sim$X, Phi, tau2 = tau2,
                    n_iter = N_ITER, burn = BURN, seed = seed)
  draws <- fit$beta[, 1]
  ci <- quantile(draws, c(0.025, 0.975))

  list(row = data.frame(
         seed = seed, mean = mean(draws), lo = ci[[1]], hi = ci[[2]],
         covers = BETA_TRUE >= ci[[1]] && BETA_TRUE <= ci[[2]],
         width = ci[[2]] - ci[[1]],
         ess = effective_size(draws),
         acf1 = stats::acf(draws, lag.max = 1, plot = FALSE)$acf[2],
         ols = unname(coef(lm(sim$data$y ~ sim$X))[2]),
         row.names = NULL),
       draws = if (keep_draws) draws else NULL)
}

cat(sprintf("beta = %.2f, confounding = %.1f, n = %d, %d folds, width %d\n",
            BETA_TRUE, CONFOUNDING, N_OBS, FOLDS, WIDTH))
cat(sprintf("%d datasets, %d Gibbs iterations (%d burn-in)\n\n",
            N_DATASETS, N_ITER, BURN))

started <- Sys.time()
results <- vector("list", N_DATASETS)
trace_draws <- NULL
for (i in seq_len(N_DATASETS)) {
  out <- one_dataset(1000 + i, keep_draws = i == 1)
  results[[i]] <- out$row
  if (i == 1) trace_draws <- out$draws
  cat("."); if (i %% 50 == 0) cat(" ", i, "\n")
  utils::flush.console()
}
cat("\n")
res <- do.call(rbind, results)
elapsed <- difftime(Sys.time(), started, units = "mins")

coverage <- mean(res$covers)
se_cov <- sqrt(0.95 * 0.05 / nrow(res))

cat(sprintf("\nelapsed: %.1f minutes\n\n", as.numeric(elapsed)))
cat(sprintf("coverage        : %.3f   (nominal 0.95, MC s.e. %.3f)\n",
            coverage, se_cov))
cat(sprintf("                  %d of %d intervals miss, expected about %.1f\n",
            sum(!res$covers), nrow(res), 0.05 * nrow(res)))
cat(sprintf("mean posterior  : %.4f  (bias %+.4f)\n",
            mean(res$mean), mean(res$mean) - BETA_TRUE))
cat(sprintf("naive OLS       : %.4f  (bias %+.4f)\n",
            mean(res$ols), mean(res$ols) - BETA_TRUE))
cat(sprintf("mean CI width   : %.4f\n", mean(res$width)))
cat(sprintf("RMSE of mean    : %.4f\n",
            sqrt(mean((res$mean - BETA_TRUE)^2))))
cat(sprintf("effective size  : median %.0f of %d kept draws\n",
            median(res$ess), N_ITER - BURN))
cat(sprintf("lag-1 autocorr  : median %.3f\n", median(res$acf1)))

within <- abs(coverage - 0.95) <= 2 * se_cov
cat(sprintf("\ncoverage is %s two Monte Carlo standard errors of nominal\n",
            if (within) "within" else "OUTSIDE"))

# --- the money plot --------------------------------------------------------

ord <- order(res$mean)
d <- res[ord, ]
col <- ifelse(d$covers, "grey30", "firebrick")

png(file.path(out_dir, "coverage_intervals.png"),
    width = 1500, height = 700, res = 130)
layout(matrix(c(1, 2), 1, 2), widths = c(1.35, 1))
par(mar = c(4.3, 4.3, 3.6, 1))

# xlim must include the naive OLS mean, or the reference line the legend
# promises is drawn off the canvas. Showing it is the point: the gap between
# it and the solid truth line is the confounding the method removes.
plot(NA, xlim = range(d$lo, d$hi, BETA_TRUE, mean(res$ols)),
     ylim = c(1, nrow(d)),
     xlab = expression(beta), ylab = "dataset (ordered by posterior mean)",
     main = sprintf("%d datasets, 95%% credible intervals\ncoverage %.1f%%, %d miss",
                    nrow(d), 100 * coverage, sum(!d$covers)))
segments(d$lo, seq_len(nrow(d)), d$hi, seq_len(nrow(d)), col = col, lwd = 1.1)
points(d$mean, seq_len(nrow(d)), pch = 16, cex = 0.35, col = col)
abline(v = BETA_TRUE, lwd = 2, col = "black")
abline(v = mean(res$ols), lwd = 1.6, lty = 3, col = "steelblue4")
legend("topleft", bty = "n", cex = 0.75,
       legend = c("covers", "misses", "true beta", "mean naive OLS"),
       col = c("grey30", "firebrick", "black", "steelblue4"),
       lty = c(1, 1, 1, 3), lwd = c(1.1, 1.1, 2, 1.6))

plot(trace_draws, type = "l", col = "grey25", lwd = 0.6,
     xlab = "iteration after burn-in", ylab = expression(beta),
     main = sprintf("trace, dataset 1\nESS %.0f of %d, lag-1 acf %.2f",
                    res$ess[1], N_ITER - BURN, res$acf1[1]))
abline(h = BETA_TRUE, col = "firebrick", lwd = 1.6, lty = 2)
abline(h = mean(trace_draws), col = "steelblue4", lwd = 1.4)

dev.off()
cat("\nwrote coverage_intervals.png to", out_dir, "\n")

# Keep the trace alongside the summaries so the figure can be redrawn without
# paying for the whole simulation again.
saveRDS(list(results = res, trace = trace_draws, beta_true = BETA_TRUE,
             n_iter = N_ITER, burn = BURN),
        file.path(out_dir, "coverage_results.rds"))
