/**
 * @file wn_grad_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the WNGrad optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_WN_GRAD_WN_GRAD_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_WN_GRAD_WN_GRAD_IMPL_HPP

// In case it hasn't been included yet.
#include "wn_grad.hpp"

namespace mlpack {
namespace optimization {

WNGrad::WNGrad(
    const double stepSize,
    const size_t batchSize,
    const size_t maxIterations,
    const double tolerance,
    const bool shuffle) :
    optimizer(stepSize,
              batchSize,
              maxIterations,
              tolerance,
              shuffle,
              WNGradUpdate())
{ /* Nothing to do. */ }

} // namespace optimization
} // namespace mlpack

#endif