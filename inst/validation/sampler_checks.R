# Visual checkpoint for the sampler object.
# Required by SESSION_PROTOCOL.md, "After the sampler object".
#
#   Rscript inst/validation/sampler_checks.R
#
# Writes one PNG into this directory.
#
# STAND-IN FEATURES. bllnn_warmup() does not exist yet, so the frozen feature
# matrix here is a random ReLU layer rather than a learned network body:
# Phi = [1, max(0, Z W + b)] with W and b drawn once and never touched again.
# That is a weaker feature map than a trained body will be, so the fit is
# pessimistic, not flattering. It also sidesteps sample splitting entirely --
# random features cannot leak the response because they never see it -- which
# means what this plot shows is the sampler's calibration, uncontaminated by
# any question about how the features were learned.

devtools::load_all(quiet = TRUE)

out_dir <- file.path("inst", "validation")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

n_draws <- 2000

sim <- sim_partial_linear(n = 400, beta = c(treat = 1.5), p_z = 5,
                          f = "friedman", confounding = 0.6, sigma = 1,
                          seed = 20260822)

# The host owns the linear part. Here we hand the block the oracle residual,
# y - X beta, so the plot isolates how well f(Z) is recovered rather than
# mixing in an error in beta. Orthogonalisation against X is not implemented
# yet; that is design rule 1 and belongs to a later session.
r <- sim$data$y - as.vector(sim$X %*% sim$beta)

set.seed(7)
Z_scaled <- scale(sim$Z)
m_hidden <- 50
W <- matrix(rnorm(ncol(Z_scaled) * m_hidden, sd = 1), ncol(Z_scaled), m_hidden)
b <- runif(m_hidden, -1, 1)
# pmax(A, 0) not pmax(0, A): with the scalar first, pmax takes its attributes
# from the scalar and silently drops the matrix dimensions.
hidden <- Z_scaled %*% W + rep(b, each = nrow(Z_scaled))
Phi <- cbind(1, pmax(hidden, 0))
colnames(Phi) <- c("intercept", paste0("h", seq_len(m_hidden)))

# tau2 chosen by a stated rule, not by looking at the plot: scale the prior so
# the implied prior variance of f(z) matches the variance of the residual.
tau2 <- var(r) / mean(rowSums(Phi^2))

mod <- bllnn_sampler(Phi, tau2 = tau2)
stopifnot(is_valid_kernel(mod))
set_response(mod, r)
set_sigma(mod, sim$sigma)

draws <- matrix(NA_real_, nrow = n_draws, ncol = nrow(Phi))
set.seed(1)
for (i in seq_len(n_draws)) draws[i, ] <- gibbs_step(mod)

# How much of f(Z) can these features represent at all? The least-squares
# projection of the truth onto span(Phi) is the best any draw could do. If the
# posterior mean's error is close to this, the error is approximation, not
# sampling, and no correctly calibrated band can cover it.
truth_raw <- sim$f_true - mean(sim$f_true)
approx_err <- sqrt(mean((lm.fit(Phi, truth_raw)$fitted.values - truth_raw)^2))

# f is identified only up to a constant, so compare on a common centring.
centre <- function(x) x - mean(x)
truth <- centre(sim$f_true)
draws_c <- t(apply(draws, 1, centre))

post_mean <- colMeans(draws_c)
lower <- apply(draws_c, 2, quantile, 0.025)
upper <- apply(draws_c, 2, quantile, 0.975)

covered <- truth >= lower & truth <= upper
coverage <- mean(covered)
band_width <- mean(upper - lower)
rmse <- sqrt(mean((post_mean - truth)^2))

ord <- order(truth)

# --- control: the same machinery on a correctly specified problem -----------
# Generate the truth from inside span(Phi), so the only error left is the one
# the posterior actually models. Coverage here must reach nominal; if it does,
# any shortfall in the panel above is misspecification and not the sampler.
set.seed(99)
w_control <- rnorm(ncol(Phi), sd = 0.3)
f_control <- as.vector(Phi %*% w_control)
set.seed(5)
r_control <- f_control + rnorm(nrow(Phi), sd = 1)

mod_c <- bllnn_sampler(Phi, tau2 = 0.09)
set_response(mod_c, r_control)
set_sigma(mod_c, 1)
draws_control <- matrix(NA_real_, nrow = n_draws, ncol = nrow(Phi))
set.seed(1)
for (i in seq_len(n_draws)) draws_control[i, ] <- gibbs_step(mod_c)

lo_c <- apply(draws_control, 2, quantile, 0.025)
hi_c <- apply(draws_control, 2, quantile, 0.975)
covered_c <- f_control >= lo_c & f_control <= hi_c
coverage_c <- mean(covered_c)
ord_c <- order(f_control)

png(file.path(out_dir, "sampler_credible_bands.png"),
    width = 1900, height = 560, res = 130)
par(mfrow = c(1, 3), mar = c(4.3, 4.3, 3.4, 1))

plot(seq_along(ord), truth[ord], type = "n",
     xlab = "observation, ordered by true f(Z)", ylab = "f(Z), centred",
     main = sprintf("95%% bands, %d draws", n_draws))
polygon(c(seq_along(ord), rev(seq_along(ord))),
        c(lower[ord], rev(upper[ord])),
        col = rgb(0.2, 0.4, 0.8, 0.22), border = NA)
lines(seq_along(ord), post_mean[ord], col = "steelblue4", lwd = 1.6)
lines(seq_along(ord), truth[ord], col = "firebrick", lwd = 1.8)
points(which(!covered[ord]), truth[ord][!covered[ord]],
       pch = 16, cex = 0.5, col = "firebrick")
legend("topleft", bty = "n", cex = 0.8,
       legend = c("true f(Z)", "posterior mean", "95% band", "outside band"),
       col = c("firebrick", "steelblue4", rgb(0.2, 0.4, 0.8, 0.5), "firebrick"),
       lty = c(1, 1, 1, NA), pch = c(NA, NA, NA, 16), lwd = c(1.8, 1.6, 6, NA))

plot(truth, post_mean, pch = 16, cex = 0.45,
     col = ifelse(covered, rgb(0.2, 0.2, 0.2, 0.5), "firebrick"),
     xlab = "true f(Z), centred", ylab = "posterior mean",
     main = sprintf("coverage %.1f%%  RMSE %.2f", 100 * coverage, rmse))
segments(truth, lower, truth, upper, col = rgb(0.2, 0.4, 0.8, 0.25))
abline(0, 1, col = "firebrick", lwd = 1.6, lty = 2)

plot(seq_along(ord_c), f_control[ord_c], type = "n",
     xlab = "observation, ordered by true f", ylab = "f, in span(Phi)",
     main = sprintf("CONTROL: correct model, coverage %.1f%%",
                    100 * coverage_c))
polygon(c(seq_along(ord_c), rev(seq_along(ord_c))),
        c(lo_c[ord_c], rev(hi_c[ord_c])),
        col = rgb(0.15, 0.55, 0.3, 0.22), border = NA)
lines(seq_along(ord_c), colMeans(draws_control)[ord_c],
      col = "darkgreen", lwd = 1.6)
lines(seq_along(ord_c), f_control[ord_c], col = "firebrick", lwd = 1.8)
points(which(!covered_c[ord_c]), f_control[ord_c][!covered_c[ord_c]],
       pch = 16, cex = 0.5, col = "firebrick")

mtext(sprintf(
  "random ReLU features (m = %d), tau2 = %.3g, sigma = %.2f  |  approximation error %.3f vs band half-width %.3f",
  ncol(Phi), tau2, sim$sigma, approx_err, band_width / 2),
  outer = TRUE, line = -1.3, cex = 0.8)
dev.off()

cat(sprintf("n = %d, features = %d, draws = %d\n",
            nrow(Phi), ncol(Phi), n_draws))
cat(sprintf("tau2 (rule-based)      : %.5g\n", tau2))
cat(sprintf("pointwise 95%% coverage : %.4f  (nominal 0.95)\n", coverage))
cat(sprintf("mean band width        : %.4f\n", band_width))
cat(sprintf("sd of true f(Z)        : %.4f\n", sd(truth)))
cat(sprintf("RMSE of posterior mean : %.4f\n", rmse))
cat(sprintf("residual sd            : %.4f\n", sd(r)))
cat(sprintf("approximation error    : %.4f  <- best possible within span(Phi)\n",
            approx_err))
cat(sprintf("band half-width        : %.4f\n", band_width / 2))
cat(sprintf("\nCONTROL (truth inside span(Phi)): coverage %.4f\n", coverage_c))
cat("\nReading: the posterior mean's error is approximation error, not\n")
cat("sampling error, and the bands only quantify uncertainty in w given that\n")
cat("f lies in span(Phi). When it does -- the control panel -- coverage is\n")
cat("nominal. The shortfall in panels 1 and 2 is the feature map, and a\n")
cat("trained body will shrink it but never remove it: frozen features always\n")
cat("mean the uncertainty is conditional on them.\n")
cat("\nwrote sampler_credible_bands.png to", out_dir, "\n")
