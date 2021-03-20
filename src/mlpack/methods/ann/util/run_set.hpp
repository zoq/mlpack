/**
 * @file methods/ann/util/run_set.hpp
 * @author Aakash Kaushik
 *
 * Definition of the RunSet() function which sets the run parameter given the 
 * run value.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_UTIL_RUN_SET_HPP
#define MLPACK_METHODS_ANN_UTIL_RUN_SET_HPP

namespace mlpack {
namespace ann /** Artificial Neural Network. */{

/**
 * Set the run parameter given the run value.
 *
 * @tparam LayerType The type of the given layer.
 * @param layer The layer to get the loss for.
 * @param run boolean value for run. 
 */
template<typename LayerType>
size_t RunSet(const LayerType& layer, const bool& run)
{
  layer->Run() = run;

  if (layer->Model().size() > 0)
  {
    for (size_t i = 0; i < layer->Model().size(); ++i)
      layer->Model()[i]->Run() = run;
  }
}

} // namespace ann
} // namespace mlpack

#endif
