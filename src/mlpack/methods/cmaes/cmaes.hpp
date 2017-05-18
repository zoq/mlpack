/**
 * @file cmaes.hpp
 * @author Marcus Edel
 *
 * CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_CMAES_CMAES_HPP
#define MLPACK_METHODS_CMAES_CMAES_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace cmaes {

/**
 * CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
 *
 * @tparam DecomposableFunctionType Decomposable objective function type to be
 *     minimized.
 */
template<typename DecomposableFunctionType>
class CMAES
{
 public:
  /**
   * Construct the CMAES optimizer with the given function and parameters.
   *
   * @param function Function to be optimized (minimized).
   */
  CMAES(DecomposableFunctionType& function);

  /**
   * Optimize the given function using CMAES.  The given
   * starting point will be modified to store the finishing point of the
   * algorithm, and the final objective value is returned.
   *
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  double Optimize(arma::mat& iterate);

  //! Get the instantiated function to be optimized.
  const DecomposableFunctionType& Function() const { return function; }
  //! Modify the instantiated function.
  DecomposableFunctionType& Function() { return function; }

 private:
  //! The instantiated function.
  DecomposableFunctionType& function;
};

} // namespace cmaes
} // namespace mlpack

// Include implementation.
#include "cmaes_impl.hpp"

#endif
