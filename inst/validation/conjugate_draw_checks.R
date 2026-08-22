# Visual checkpoint for the conjugate last-layer draw.
# Required by SESSION_PROTOCOL.md, "After the conjugate draw".
#
# Passing tests and correct statistics are not the same thing. Run this and
# look at the output. Regenerate with:
#
#   Rscript inst/validation/conjugate_draw_checks.R
#
# Writes three PNGs into this directory.

devtools::load_all(quiet = TRUE)

out_dir <- file.path("inst", "validation")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

n_draws <- 50000

# 95% contour of a bivariate normal. V = R'R with R upper triangular, so
# mu + sqrt(q) * t(R) %*% u traces the contour as u runs round the unit circle.
ellipse_points <- function(mu, V, level = 0.95, n_pts = 400) {
  q <- stats::qchisq(level, df = 2)
  R <- chol(V)
  theta <- seq(0, 2 * pi, length.out = n_pts)
  u <- rbind(cos(theta), sin(theta))
  t(mu + sqrt(q) * t(R) %*% u)
}

draw_many <- function(Phi, r, sigma, tau2, n_draws, seed = 1) {
  set.seed(seed)
  m <- ncol(Phi)
  out <- matrix(NA_real_, nrow = n_draws, ncol = m)
  for (i in seq_len(n_draws)) out[i, ] <- conjugate_draw(Phi, r, sigma, tau2)
  out
}

# --- design 1: the same setup the Monte Carlo test uses ---------------------

set.seed(20260822)
n <- 40
m <- 3
Phi <- matrix(rnorm(n * m), n, m)
r <- as.vector(Phi %*% c(1, -2, 0.5)) + rnorm(n, sd = 0.7)
sigma <- 0.7
tau2 <- 2

post <- conjugate_moments(Phi, r, sigma, tau2)
draws <- draw_many(Phi, r, sigma, tau2, n_draws)

# --- figure 1: marginals against the analytic normal density ----------------

png(file.path(out_dir, "conjugate_marginals.png"),
    width = 1500, height = 520, res = 130)
par(mfrow = c(1, 3), mar = c(4.2, 4.2, 3.2, 1))
for (j in seq_len(m)) {
  hist(draws[, j], breaks = 80, freq = FALSE, border = NA, col = "grey80",
       main = sprintf("w[%d]", j),
       xlab = sprintf("analytic N(%.3f, %.4f)", post$mean[j], post$cov[j, j]))
  grid_x <- seq(min(draws[, j]), max(draws[, j]), length.out = 400)
  lines(grid_x,
        dnorm(grid_x, mean = post$mean[j], sd = sqrt(post$cov[j, j])),
        lwd = 2, col = "firebrick")
  abline(v = post$mean[j], lty = 2, col = "firebrick")
}
mtext(sprintf("%s draws vs analytic marginals", format(n_draws, big.mark = ",")),
      outer = TRUE, line = -1.4, cex = 0.9)
dev.off()

# --- figure 2: joint structure, all pairs, against the 95% ellipse ----------
# This is the one that matters. Marginals can look perfect while the
# correlation structure is wrong, and only the joint view shows it.

pairs_idx <- list(c(1, 2), c(1, 3), c(2, 3))
sub <- sample.int(n_draws, 8000)

png(file.path(out_dir, "conjugate_joint.png"),
    width = 1500, height = 540, res = 130)
par(mfrow = c(1, 3), mar = c(4.2, 4.2, 3.2, 1))
for (p in pairs_idx) {
  i <- p[1]; j <- p[2]
  V2 <- post$cov[p, p]
  emp_r <- cor(draws[, i], draws[, j])
  ana_r <- V2[1, 2] / sqrt(V2[1, 1] * V2[2, 2])

  plot(draws[sub, i], draws[sub, j], pch = 16, cex = 0.2,
       col = rgb(0.2, 0.2, 0.2, 0.12), asp = 1,
       xlab = sprintf("w[%d]", i), ylab = sprintf("w[%d]", j),
       main = sprintf("cor: analytic %.3f / empirical %.3f", ana_r, emp_r))
  ell <- ellipse_points(post$mean[p], V2)
  lines(ell[, 1], ell[, 2], lwd = 2, col = "firebrick")
  points(post$mean[i], post$mean[j], pch = 3, lwd = 2, col = "firebrick")
}
mtext("95% analytic ellipse over the draws", outer = TRUE, line = -1.4, cex = 0.9)
dev.off()

# --- figure 3: what the joint plot is for ----------------------------------
# A collinear design, where the posterior correlation is strong enough that a
# diagonal-only draw is obviously wrong despite having exact marginals.

set.seed(5)
n2 <- 60
phi1 <- rnorm(n2)
Phi_c <- cbind(phi1, 0.97 * phi1 + rnorm(n2, sd = 0.24))
colnames(Phi_c) <- c("f1", "f2")
r_c <- as.vector(Phi_c %*% c(1.5, -0.5)) + rnorm(n2, sd = 0.5)

post_c <- conjugate_moments(Phi_c, r_c, sigma = 0.5, tau2 = 10)
draws_c <- draw_many(Phi_c, r_c, sigma = 0.5, tau2 = 10, n_draws, seed = 2)

# The failure mode: correct marginal variances, independent draws. Every
# histogram would pass; the joint distribution is badly wrong.
set.seed(3)
draws_diag <- cbind(
  rnorm(n_draws, post_c$mean[1], sqrt(post_c$cov[1, 1])),
  rnorm(n_draws, post_c$mean[2], sqrt(post_c$cov[2, 2]))
)

ell_c <- ellipse_points(post_c$mean, post_c$cov)
ana_r <- post_c$cov[1, 2] / sqrt(post_c$cov[1, 1] * post_c$cov[2, 2])
xlim <- range(ell_c[, 1], draws_diag[sub, 1])
ylim <- range(ell_c[, 2], draws_diag[sub, 2])

png(file.path(out_dir, "conjugate_cholesky_orientation.png"),
    width = 1100, height = 560, res = 130)
par(mfrow = c(1, 2), mar = c(4.2, 4.2, 3.2, 1))

plot(draws_c[sub, 1], draws_c[sub, 2], pch = 16, cex = 0.2,
     col = rgb(0.2, 0.2, 0.2, 0.12), xlim = xlim, ylim = ylim,
     xlab = "w[f1]", ylab = "w[f2]",
     main = sprintf("conjugate_draw()  cor = %.3f", cor(draws_c[, 1], draws_c[, 2])))
lines(ell_c[, 1], ell_c[, 2], lwd = 2, col = "firebrick")

plot(draws_diag[sub, 1], draws_diag[sub, 2], pch = 16, cex = 0.2,
     col = rgb(0.2, 0.2, 0.2, 0.12), xlim = xlim, ylim = ylim,
     xlab = "w[f1]", ylab = "w[f2]",
     main = sprintf("diagonal-only  cor = %.3f", cor(draws_diag[, 1], draws_diag[, 2])))
lines(ell_c[, 1], ell_c[, 2], lwd = 2, col = "firebrick")

mtext(sprintf("identical marginals, analytic cor = %.3f", ana_r),
      outer = TRUE, line = -1.4, cex = 0.9)
dev.off()

# --- numbers to read alongside the plots -----------------------------------

cat("design 1 (m = 3)\n")
cat("  analytic mean :", format(post$mean, digits = 5), "\n")
cat("  empirical mean:", format(colMeans(draws), digits = 5), "\n")
cat("  max |cov error|:", max(abs(cov(draws) - post$cov)), "\n")
cat("  analytic cor  :", format(cov2cor(post$cov)[upper.tri(post$cov)], digits = 4), "\n")
cat("  empirical cor :", format(cor(draws)[upper.tri(post$cov)], digits = 4), "\n\n")

cat("design 2 (collinear, m = 2)\n")
cat("  analytic cor            :", format(ana_r, digits = 4), "\n")
cat("  conjugate_draw() cor    :", format(cor(draws_c[, 1], draws_c[, 2]), digits = 4), "\n")
cat("  diagonal-only cor       :", format(cor(draws_diag[, 1], draws_diag[, 2]), digits = 4), "\n")
cat("  marginal sd analytic    :", format(sqrt(diag(post_c$cov)), digits = 4), "\n")
cat("  marginal sd diag-only   :", format(apply(draws_diag, 2, sd), digits = 4), "\n")
cat("\nwrote 3 PNGs to", out_dir, "\n")
