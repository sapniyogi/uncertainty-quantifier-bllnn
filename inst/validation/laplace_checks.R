# What the Laplace approximation costs, measured against the exact posterior.
#
#   Rscript inst/validation/laplace_checks.R      (roughly 35 minutes)
#
# Writes one PNG into this directory. Most of that time is the yardstick, not
# the method: the Polya-Gamma sampler is pure R and its cost is linear in both
# the number of draws and n, so the n = 1000 cell alone is a quarter of the
# run. Trim `n_grid` if a faster answer will do -- the trend is visible by
# n = 500.
#
# The Laplace path is the only approximate method in the package, so the
# question it has to answer is not "does it run" but "how wrong is it, and in
# which direction". Both parts matter: an approximation that is merely close is
# useful, while one that is close but systematically overconfident is a trap,
# because the error lands in exactly the quantity -- interval width -- that the
# package exists to get right.
#
# The yardstick is the Polya-Gamma sampler, which targets the exact logistic
# posterior. A long run of it is ground truth, not another approximation, so
# the gap measured here is the Laplace error alone up to Monte Carlo noise.
#
# What the figure shows: both errors shrink with n, which is Bernstein-von
# Mises made visible -- the posterior becomes Gaussian, and a Gaussian fitted
# at its mode becomes exact. Measured here, the mode-to-mean gap falls from
# 0.45 posterior sd at n = 60 to 0.14 at n = 1000, and the interval width goes
# from 0.95 of the exact width to 1.00 by n = 500. The approximation is
# therefore a small-sample concern and not an asymptotic one; where it costs
# something, it costs it in the direction of overconfidence.

devtools::load_all(quiet = TRUE)

out_dir <- file.path("inst", "validation")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

tau2 <- 4
n_draws <- 8000
burn <- n_draws / 4

# Exact posterior for the logistic model, by long-running Polya-Gamma.
exact_posterior <- function(Phi, y, n_draws, burn) {
  mod <- bllnn_sampler(Phi, tau2 = tau2, posterior = "polyagamma")
  set_response(mod, y)
  W <- matrix(0, n_draws, ncol(Phi))
  for (i in seq_len(n_draws)) {
    gibbs_step(mod)
    W[i, ] <- mod$w
  }
  W[-seq_len(burn), , drop = FALSE]
}

make_case <- function(n, m, seed) {
  set.seed(seed)
  Phi <- matrix(stats::rnorm(n * m), n, m)
  w0 <- stats::rnorm(m)
  list(Phi = Phi, y = stats::rbinom(n, 1, stats::plogis(
    as.vector(Phi %*% w0))))
}

# --- panel A: the marginals at a small sample size --------------------------

case <- make_case(n = 80, m = 6, seed = 9)
lap <- laplace_moments(case$Phi, case$y, tau2 = tau2, family = "binomial")
W <- exact_posterior(case$Phi, case$y, n_draws, burn)

cat(sprintf("[laplace] n=80  m=6  exact draws kept: %d\n", nrow(W)))

# --- panel B: how the gap closes with n -------------------------------------

n_grid <- c(60, 120, 250, 500, 1000)
gap <- numeric(length(n_grid))
ratio <- numeric(length(n_grid))

for (i in seq_along(n_grid)) {
  ci <- make_case(n = n_grid[i], m = 4, seed = 100 + i)
  li <- laplace_moments(ci$Phi, ci$y, tau2 = tau2, family = "binomial")
  Wi <- exact_posterior(ci$Phi, ci$y, n_draws, burn)

  sd_exact <- apply(Wi, 2, stats::sd)
  gap[i] <- max(abs(li$mean - colMeans(Wi)) / sd_exact)
  ratio[i] <- mean(sqrt(diag(li$cov)) / sd_exact)

  cat(sprintf("[laplace] n=%4d  gap %.3f sd  sd ratio %.3f\n",
              n_grid[i], gap[i], ratio[i]))
}

# --- figure -----------------------------------------------------------------

png(file.path(out_dir, "laplace_vs_exact.png"), width = 1100, height = 760,
    res = 110)
on.exit(grDevices::dev.off(), add = TRUE)

layout(matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, byrow = TRUE))
par(mar = c(4, 4, 3, 1))

exact_col <- grDevices::rgb(0.15, 0.35, 0.65, 0.5)
lap_col <- grDevices::rgb(0.85, 0.25, 0.15, 1)

for (j in 1:4) {
  d <- stats::density(W[, j])
  grid_j <- seq(min(d$x), max(d$x), length.out = 400)
  lap_d <- stats::dnorm(grid_j, lap$mean[j], sqrt(lap$cov[j, j]))

  plot(d, main = sprintf("weight %d", j), xlab = "", ylab = "density",
       ylim = c(0, max(d$y, lap_d) * 1.05), col = NA)
  polygon(d, col = exact_col, border = NA)
  lines(grid_j, lap_d, col = lap_col, lwd = 2)
  if (j == 1) {
    legend("topright", bty = "n", cex = 0.8,
           legend = c("exact (Polya-Gamma)", "Laplace"),
           fill = c(exact_col, NA), border = c(NA, NA),
           lty = c(NA, 1), lwd = c(NA, 2), col = c(NA, lap_col))
  }
}

plot(n_grid, gap, type = "b", pch = 19, log = "x",
     ylim = c(0, max(gap) * 1.15), col = lap_col, lwd = 2,
     main = "mode vs posterior mean", xlab = "n",
     ylab = "largest gap, in posterior sd")
abline(h = 0, col = "grey60", lty = 2)

plot(n_grid, ratio, type = "b", pch = 19, log = "x",
     ylim = range(c(ratio, 1)) + c(-0.02, 0.02), col = lap_col, lwd = 2,
     main = "interval width", xlab = "n",
     ylab = "Laplace sd / exact sd")
abline(h = 1, col = "grey40", lty = 2)
text(n_grid[length(n_grid)], 1, "exact", pos = 3, cex = 0.75, col = "grey40")

cat(sprintf("[laplace] wrote %s\n",
            file.path(out_dir, "laplace_vs_exact.png")))
