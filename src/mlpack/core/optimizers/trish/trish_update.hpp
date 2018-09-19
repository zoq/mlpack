/**
 * @file trish_update.hpp
 * @author Marcus Edel
 *
 * TRish update rule for the TRish method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_TRISH_TRISH_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_TRISH_TRISH_UPDATE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * TRish is a stochastic Trust Region algorithm based on a careful step
 * normalization.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Wu2018,
 *   author  = {{Curtis}, F.~E. and {Scheinberg}, K. and {Shi}, R.},
 *   title   = {A Stochastic Trust Region Algorithm Based on Careful Step
 *              Normalization},
 *   journal = {ArXiv e-prints},
 *   year    = {2017},
 *   url     = {https://arxiv.org/abs/1712.10277},
 * }
 * @endcode
 */
class TRishUpdate
{
 public:
  /**
   * Construct the TRish update policy.
   *
   * @param gamma1 Stepsize normalization term.
   * @param gamma2 Stepsize normalization term.
   */
  TRishUpdate(const double gamma1 = 22.90, const double gamma2 = 2.863) :
      gamma1(gamma1), gamma2(gamma2)
  {
    // Nothing to do here.
  }

  /**
   * The Initialize method is called by SGD Optimizer method before the start of
   * the iteration update process.
   *
   * @param rows Number of rows in the gradient matrix.
   * @param cols Number of columns in the gradient matrix.
   */
  void Initialize(const size_t /* rows */, const size_t /* cols */)
  {
    // Nothing to do here.
  }

  /**
   * Update step for TRish.
   *
   * @param iterate Parameters that minimize the function.
   * @param stepSize Step size to be used for the given iteration.
   * @param gradient The gradient matrix.
   */
  void Update(arma::mat& iterate,
              const double stepSize,
              const arma::mat& gradient)
  {
    const double gn = arma::norm(gradient);

    if (!(gn < 0) && (gn < 1 / gamma1))
    {
      iterate -= gamma1 * stepSize * gradient;
    }
    else if (!(gn < 1 / gamma1) && (gn < 1 / gamma2))
    {
      iterate -= stepSize * gradient / gn;
    }
    else
    {
      iterate -= gamma2 * stepSize * gradient;
    }    
  }

  //! Get the first Stepsize normalization term.
  double Gamma1() const { return gamma1; }
  //! Modify the first Stepsize normalization term.
  double& Gamma1() { return gamma1; }

  //! Get the second Stepsize normalization term.
  double Gamma2() const { return gamma2; }
  //! Modify the second Stepsize normalization term.
  double& Gamma2() { return gamma2; }

 private:
  //! First stepsize normalization term.
  double gamma1;

  //! Second stepsize normalization term.
  double gamma2;
};

} // namespace optimization
} // namespace mlpack

#endif