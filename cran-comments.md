# cran-comments

## Submission

First submission of `bllnn`.

## Test environments

* local: Windows 11, R 4.6.1, Rtools45 (gcc/g++ 14.3.0)
* (add before submitting: win-builder devel and release, and at least one
  Linux and one macOS platform, e.g. via R-hub or GitHub Actions)

## R CMD check results

0 errors | 0 warnings | 0 notes

On first submission CRAN will additionally raise the standard NOTE for a new
submission ("New submission"), which needs no action.

## Notes for the reviewer

The package contains compiled code (`RcppArmadillo`) whose only current use is
a single routine. The numeric core is deliberately plain R for this release:
the R implementation is the reference specification that a later C++ port will
be tested against, and shipping the port before that reference is stable would
invert the relationship. `src/` and the linking setup are present so the port
can land without a structural change.

Vignettes build in roughly 75 seconds in total. One of them
(`lalonde.Rmd`) uses the `lalonde` dataset from `MatchIt`, which is a
Suggests-level dependency; the vignette guards on `requireNamespace()` and
renders an explanatory note if the package is absent, so it builds either way.
No data is bundled with this package.

## Downstream dependencies

None; this is a first submission.
