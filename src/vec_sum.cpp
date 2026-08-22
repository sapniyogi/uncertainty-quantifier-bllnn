#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

//' Sum of a numeric vector
//'
//' A toolchain smoke test, not a statistical routine. It exists to prove that
//' the C++ compiler, the Rcpp/RcppArmadillo headers, and the R binding all
//' work end to end on this machine. Delete it once a real numeric core lands.
//'
//' @param x A numeric vector.
//'
//' @return A length-one numeric vector: the sum of `x`. The sum of an empty
//'   vector is `0`.
//'
//' @examples
//' vec_sum(c(1, 2, 3))
//'
//' @export
// [[Rcpp::export]]
double vec_sum(const arma::vec& x) {
  return arma::accu(x);
}
