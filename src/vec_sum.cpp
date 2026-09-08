#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

//' Sum of a numeric vector
//'
//' A toolchain smoke test, not a statistical routine, and deliberately not
//' exported: it exists to prove that the C++ compiler, the Rcpp/RcppArmadillo
//' headers, the Makevars link line and the R binding all work end to end, so
//' that the numeric core can be ported without first discovering the build is
//' broken. Duplicating `sum()` in the public API would be worse than useless.
//'
//' It stays until the port replaces it with something real, at which point it
//' and its test should go.
//'
//' @param x A numeric vector.
//'
//' @return A length-one numeric vector: the sum of `x`. The sum of an empty
//'   vector is `0`.
//'
//' @noRd
// [[Rcpp::export]]
double vec_sum(const arma::vec& x) {
  return arma::accu(x);
}
