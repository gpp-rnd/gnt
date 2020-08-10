=======
History
=======

0.1.0 (2020-06-29)
------------------

* First release on PyPI.

0.1.1 (2020-07-09)
------------------

* Automated release

0.1.2 (2020-07-09)
------------------

* Travis bug fix

0.2.0 (2020-07-13)
------------------

* Add columns to guide and gene output for base LFC of pairs
* Check inputs, removing guides without the right number of pairs or control pairs


0.2.1 (2020-07-13)
------------------

* Update basic usage notebook

0.2.2 (2020-07-15)
------------------

* Aggregate guide scores that are in data multiple times in different orientations

0.2.3 (2020-07-15)
------------------

* Deduplicate repeat guide pairs in anchor df

0.2.4 (2020-07-22)
------------------

* Add model coefficients to guide residual ouput
* Update delta-LFC functions

0.2.5 (2020-07-27)
------------------

* Add base LFC to dLFC output

0.3.0 (2020-08-10)
------------------

* Added spline, fixed slope and quadratic models for calculating guide residuals
* Combined z-scores by square root of sample size rather than re-calculating z-scores
