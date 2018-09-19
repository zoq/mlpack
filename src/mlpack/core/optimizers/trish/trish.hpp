/**
 * @file trish.hpp
 * @author Marcus Edel
 *
 * TRish is a stochastic Trust Region algorithm based on a careful step
 * normalization.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_TRISH_TRISH_HPP
#define MLPACK_CORE_OPTIMIZERS_TRISH_TRISH_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>

#include "trish_update.hpp"

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
 *
 * For TRish to work, a DecomposableFunctionType template parameter is
 * required. This class must implement the following function:
 *
 *   size_t NumFunctions();
 *   double Evaluate(const arma::mat& coordinates,
 *                   const size_t i,
 *                   const size_t batchSize);
 *   void Gradient(const arma::mat& coordinates,
 *                 const size_t i,
 *                 arma::mat& gradient,
 *                 const size_t batchSize);
 *
 * NumFunctions() should return the number of functions (\f$n\f$), and in the
 * other two functions, the parameter i refers to which individual function (or
 * gradient) is being evaluated.  So, for the case of a data-dependent function,
 * such as NCA (see mlpack::nca::NCA), NumFunctions() should return the number
 * of points in the dataset, and Evaluate(coordinates, 0) will evaluate the
 * objective function on the first point in the dataset (presumably, the dataset
 * is held internally in the DecomposableFunctionType).
 */
class TRish
{
 public:
  /**
   * Construct the TRish optimizer with the given function and parameters. The
   * defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.  The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param stepSize Step size for each iteration.
   * @param batchSize Number of points to process in a single step.
   * @param gamma1 Stepsize normalization term.
   * @param gamma2 Stepsize normalization term.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *        limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *        function is visited in linear order.
   */
  TRish(const double stepSize = 0.4471,
        const size_t batchSize = 32,
        const double gamma1 = 22.90,
        const double gamma2 = 2.863,
        const size_t maxIterations = 100000,
        const double tolerance = 1e-5,
        const bool shuffle = true);

  /**
   * Optimize the given function using TRish. The given starting point will
   * be modified to store the finishing point of the algorithm, and the final
   * objective value is returned.
   *
   * @tparam DecomposableFunctionType Type of the function to be optimized.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  template<typename DecomposableFunctionType>
  double Optimize(DecomposableFunctionType& function, arma::mat& iterate)
  {
    return optimizer.Optimize(function, iterate);
  }

  //! Get the step size.
  double StepSize() const { return optimizer.StepSize(); }
  //! Modify the step size.
  double& StepSize() { return optimizer.StepSize(); }

  //! Get the batch size.
  size_t BatchSize() const { return optimizer.BatchSize(); }
  //! Modify the batch size.
  size_t& BatchSize() { return optimizer.BatchSize(); }

  //! Get the first Stepsize normalization term.
  double Gamma1() const { return optimizer.UpdatePolicy().Gamma1(); }
  //! Modify the first Stepsize normalization term.
  double& Gamma1() { return optimizer.UpdatePolicy().Gamma1(); }

  //! Get the second Stepsize normalization term.
  double Gamma2() const { return optimizer.UpdatePolicy().Gamma2(); }
  //! Modify the second Stepsize normalization term.
  double& Gamma2() { return optimizer.UpdatePolicy().Gamma2(); }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return optimizer.MaxIterations(); }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return optimizer.MaxIterations(); }

  //! Get the tolerance for termination.
  double Tolerance() const { return optimizer.Tolerance(); }
  //! Modify the tolerance for termination.
  double& Tolerance() { return optimizer.Tolerance(); }

  //! Get whether or not the individual functions are shuffled.
  bool Shuffle() const { return optimizer.Shuffle(); }
  //! Modify whether or not the individual functions are shuffled.
  bool& Shuffle() { return optimizer.Shuffle(); }

 private:
  //! The TRish update policy.
  SGD<TRishUpdate> optimizer;
};

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "trish_impl.hpp"

#endif