test_that("vec_sum returns the closed-form answer", {
  expect_equal(vec_sum(c(1, 2, 3)), 6)
  expect_equal(vec_sum(c(1, 2, 3, 4.5)), 10.5)
  expect_equal(vec_sum(c(-1.5, 1.5)), 0)
})

test_that("vec_sum agrees with base::sum", {
  set.seed(1)
  x <- rnorm(1000)
  expect_equal(vec_sum(x), sum(x))
})

test_that("vec_sum handles the empty vector", {
  expect_equal(vec_sum(numeric(0)), 0)
})
