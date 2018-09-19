/**
 * @file trish_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the TRish optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_TRISH_TRISH_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_TRISH_TRISH_IMPL_HPP

// In case it hasn't been included yet.
#include "trish.hpp"

namespace mlpack {
namespace optimization {

TRish::TRish(
    const double stepSize,
    const size_t batchSize,
    const double gamma1,
    const double gamma2,
    const size_t maxIterations,
    const double tolerance,
    const bool shuffle) :
    optimizer(stepSize,
              batchSize,
              maxIterations,
              tolerance,
              shuffle,
              TRishUpdate(gamma1, gamma2))
{ /* Nothing to do. */ }

} // namespace optimization
} // namespace mlpack

#endif