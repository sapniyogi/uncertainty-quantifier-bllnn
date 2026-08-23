sampler_case <- function() {
  set.seed(20260822)
  n <- 60
  m <- 4
  Phi <- matrix(rnorm(n * m), n, m)
  colnames(Phi) <- paste0("f", seq_len(m))
  list(Phi = Phi,
       r = as.vector(Phi %*% c(1, -2, 0.5, 0.3)) + rnorm(n, sd = 0.6),
       sigma = 0.6, tau2 = 5, n = n, m = m)
}

ready_sampler <- function(case = sampler_case()) {
  mod <- bllnn_sampler(case$Phi, case$tau2)
  set_response(mod, case$r)
  set_sigma(mod, case$sigma)
  mod
}

# --- a. the most important test in the file ---------------------------------

test_that("consecutive draws with identical inputs differ", {
  mod <- ready_sampler()

  f1 <- gibbs_step(mod)
  f2 <- gibbs_step(mod)

  # If these were ever equal the object would be an estimator wearing a
  # sampler's interface, and every credible interval built on it would be
  # degenerate. Nothing else in this file matters if this fails.
  expect_false(identical(f1, f2))
  expect_false(isTRUE(all.equal(f1, f2)))
  expect_gt(max(abs(f1 - f2)), 1e-8)

  # ... but they are draws of the same object, not noise: both are finite,
  # length n, and near the analytic fitted values.
  expect_length(f1, 60)
  expect_length(f2, 60)
  expect_true(all(is.finite(c(f1, f2))))
})

test_that("the weight draw stays accessible on the object", {
  case <- sampler_case()
  mod <- ready_sampler(case)

  expect_null(mod$w)
  f <- gibbs_step(mod)

  expect_length(mod$w, case$m)
  expect_named(mod$w, colnames(case$Phi))
  # The returned vector really is Phi %*% w.
  expect_equal(f, as.vector(case$Phi %*% mod$w))
  expect_equal(mod$n_steps, 1L)

  gibbs_step(mod)
  expect_equal(mod$n_steps, 2L)
})

# --- b. reproducibility -----------------------------------------------------

test_that("a fixed seed reproduces the sequence of draws", {
  mod <- ready_sampler()

  set.seed(42)
  first <- list(gibbs_step(mod), gibbs_step(mod), gibbs_step(mod))
  set.seed(42)
  second <- list(gibbs_step(mod), gibbs_step(mod), gibbs_step(mod))

  expect_identical(first, second)

  set.seed(43)
  third <- gibbs_step(mod)
  expect_false(isTRUE(all.equal(first[[1]], third)))
})

# --- c. convergence to the analytic fitted values ---------------------------

expect_mean_converges <- function(n_steps) {
  case <- sampler_case()
  mod <- ready_sampler(case)

  post <- conjugate_moments(case$Phi, case$r, case$sigma, case$tau2)
  analytic_f <- as.vector(case$Phi %*% post$mean)

  # Cov(Phi w) = Phi V Phi', so se of the averaged draw is its diagonal over N.
  se_f <- sqrt(diag(case$Phi %*% post$cov %*% t(case$Phi)) / n_steps)

  set.seed(1)
  total <- numeric(case$n)
  for (i in seq_len(n_steps)) total <- total + gibbs_step(mod)
  empirical_f <- total / n_steps

  cat(sprintf("[sampler mean N=%d] max|err|=%.5f  max err/se=%.2f\n",
              n_steps, max(abs(empirical_f - analytic_f)),
              max(abs(empirical_f - analytic_f) / se_f)))

  expect_true(all(abs(empirical_f - analytic_f) < 5 * se_f))
}

test_that("averaging 20,000 draws converges to Phi %*% posterior mean", {
  skip_on_cran()
  expect_mean_converges(20000)
})

test_that("averaging 2,000 draws converges to Phi %*% posterior mean", {
  expect_mean_converges(2000)
})

# --- d. reference semantics -------------------------------------------------

test_that("set_response changes later draws without the caller reassigning", {
  case <- sampler_case()
  mod <- ready_sampler(case)

  set.seed(9)
  before <- gibbs_step(mod)

  # Deliberately no `mod <- ...` here. That is the whole point.
  set.seed(4)
  new_r <- rnorm(case$n)
  set_response(mod, new_r)

  set.seed(9)
  after <- gibbs_step(mod)

  # Same seed, so any difference comes from the residual and not the RNG.
  expect_false(isTRUE(all.equal(before, after)))
  expect_equal(mod$r, new_r)
  expect_equal(mod$Phi_r, crossprod(case$Phi, new_r))
})

test_that("set_sigma changes later draws without the caller reassigning", {
  mod <- ready_sampler()

  set.seed(9)
  before <- gibbs_step(mod)
  set_sigma(mod, 2.5)
  set.seed(9)
  after <- gibbs_step(mod)

  expect_equal(mod$sigma, 2.5)
  expect_false(isTRUE(all.equal(before, after)))
})

test_that("the setters return the object invisibly", {
  case <- sampler_case()
  mod <- bllnn_sampler(case$Phi, case$tau2)

  expect_invisible(set_response(mod, case$r))
  expect_invisible(set_sigma(mod, case$sigma))
  expect_identical(set_sigma(mod, case$sigma), mod)
})

# --- e/f. unset and invalid state -------------------------------------------

test_that("gibbs_step refuses to run before the state is set", {
  case <- sampler_case()
  mod <- bllnn_sampler(case$Phi, case$tau2)

  expect_error(gibbs_step(mod), "set_response")

  set_response(mod, case$r)
  expect_error(gibbs_step(mod), "set_sigma")

  set_sigma(mod, case$sigma)
  expect_no_error(gibbs_step(mod))
})

test_that("set_response validates the residual length", {
  case <- sampler_case()
  mod <- bllnn_sampler(case$Phi, case$tau2)

  expect_error(set_response(mod, rnorm(case$n + 1)), "length 61")
  expect_error(set_response(mod, rnorm(case$n - 1)), "length 59")
  expect_error(set_response(mod, numeric(0)), "length 0")
  expect_error(set_response(mod, replace(case$r, 1, NA)), "no NAs")
  expect_error(set_response(mod, as.character(case$r)), "numeric")

  # A rejected residual must not leave partial state behind.
  expect_null(mod$r)
  expect_null(mod$Phi_r)
})

test_that("set_sigma validates its argument", {
  case <- sampler_case()
  mod <- bllnn_sampler(case$Phi, case$tau2)

  expect_error(set_sigma(mod, 0), "positive")
  expect_error(set_sigma(mod, -1), "positive")
  expect_error(set_sigma(mod, c(1, 2)), "single positive")
  expect_error(set_sigma(mod, NA_real_), "positive")
})

# --- g. the validity guard --------------------------------------------------

test_that("is_valid_kernel reports the combination and gibbs_step enforces it", {
  case <- sampler_case()

  good <- bllnn_sampler(case$Phi, case$tau2, "conjugate", "frozen")
  expect_true(is_valid_kernel(good))

  bad <- bllnn_sampler(case$Phi, case$tau2, "bootstrap", "frozen")
  expect_false(is_valid_kernel(bad))
  expect_match(attr(is_valid_kernel(bad), "reason"), "not a conditional",
               ignore.case = TRUE)

  set_response(bad, case$r)
  set_sigma(bad, case$sigma)

  # The guard fires before anything is drawn, and names the combination.
  expect_error(gibbs_step(bad), "not a valid Gibbs transition kernel")
  expect_error(gibbs_step(bad), "bootstrap")

  # force = TRUE is the documented escape hatch and must actually work.
  expect_no_error(f <- gibbs_step(bad, force = TRUE))
  expect_length(f, case$n)

  expect_error(gibbs_step(good, force = "yes"), "TRUE or FALSE")
})

test_that("every invalid combination in the table is refused", {
  case <- sampler_case()
  tbl <- valid_kernels()

  expect_true(any(tbl$valid))
  expect_equal(sum(tbl$valid), 1L)

  for (i in seq_len(nrow(tbl))) {
    mod <- bllnn_sampler(case$Phi, case$tau2, tbl$posterior[i], tbl$features[i])
    # as.logical() drops the `reason` attribute, which is documented output
    # rather than something to compare against the bare table column.
    expect_equal(as.logical(is_valid_kernel(mod)), tbl$valid[i])

    set_response(mod, case$r)
    set_sigma(mod, case$sigma)
    if (tbl$valid[i]) {
      expect_no_error(gibbs_step(mod))
    } else {
      expect_error(gibbs_step(mod), "not a valid Gibbs transition kernel")
    }
  }
})

test_that("unknown posterior or feature names are refused at construction", {
  case <- sampler_case()

  expect_error(bllnn_sampler(case$Phi, case$tau2, "magic"), "`posterior` must")
  expect_error(bllnn_sampler(case$Phi, case$tau2, features = "thawed"),
               "`features` must")
  expect_error(bllnn_sampler(case$Phi, 0), "positive")
  expect_error(bllnn_sampler(as.data.frame(case$Phi), 1), "numeric matrix")
})

# --- the precomputation contract --------------------------------------------

test_that("everything derivable from Phi alone is computed at construction", {
  case <- sampler_case()
  mod <- bllnn_sampler(case$Phi, case$tau2)

  expect_equal(mod$cross, crossprod(case$Phi))
  expect_equal(mod$prior_precision, diag(case$m) / case$tau2)
  expect_equal(mod$n, case$n)
  expect_equal(mod$m, case$m)

  # Phi'r belongs to the response, so it appears only once one is set.
  expect_null(mod$Phi_r)
  set_response(mod, case$r)
  expect_equal(mod$Phi_r, crossprod(case$Phi, case$r))
})

test_that("gibbs_step draws from the same posterior as the reference function", {
  case <- sampler_case()
  mod <- ready_sampler(case)

  # The sampler reaches the moments through cached pieces and
  # conjugate_moments() computes them from scratch. Identical output is what
  # says the precomputation changed the bookkeeping and not the mathematics.
  from_cache <- conjugate_moments_core(mod$cross, mod$Phi_r,
                                       mod$prior_precision, mod$sigma)
  from_scratch <- conjugate_moments(case$Phi, case$r, case$sigma, case$tau2)

  expect_equal(from_cache$mean, from_scratch$mean)
  expect_equal(from_cache$cov, from_scratch$cov)
  expect_equal(from_cache$precision, from_scratch$precision)

  # And the draws agree exactly given the same random stream.
  set.seed(77)
  from_sampler <- gibbs_step(mod)
  set.seed(77)
  from_reference <- conjugate_draw(case$Phi, case$r, case$sigma, case$tau2)

  expect_equal(mod$w, from_reference)
  expect_equal(from_sampler, as.vector(case$Phi %*% from_reference))
})

test_that("the object survives a wide feature matrix", {
  set.seed(3)
  Phi <- matrix(rnorm(5 * 9), 5, 9)
  mod <- bllnn_sampler(Phi, tau2 = 1)
  set_response(mod, rnorm(5))
  set_sigma(mod, 1)

  f <- gibbs_step(mod)
  expect_length(f, 5)
  expect_length(mod$w, 9)
  expect_true(all(is.finite(f)))
})

test_that("print reports the state without evaluating the draw", {
  case <- sampler_case()
  mod <- bllnn_sampler(case$Phi, case$tau2)

  expect_output(print(mod), "bllnn_sampler")
  expect_output(print(mod), "not set")
  expect_output(print(mod), "conjugate")

  bad <- bllnn_sampler(case$Phi, case$tau2, "vi", "frozen")
  expect_output(print(bad), "NO")
})
