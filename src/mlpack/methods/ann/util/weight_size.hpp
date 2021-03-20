/**
 * @file methods/ann/util/weight_size.hpp
 * @author Aakash Kaushik
 *
 * Definition of the WeightSize() function which returns the number of weights 
 * of the layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_UTIL_WEIGHT_SIZE_HPP
#define MLPACK_METHODS_ANN_UTIL_WEIGHT_SIZE_HPP

namespace mlpack {
namespace ann /** Artificial Neural Network. */{

/**
 * Get the number of weights from the given layer.
 *
 * @tparam LayerType The type of the given layer.
 * @param layer The layer to get the loss for.
 * @return The number of weights.
 */
template<typename LayerType>
size_t WeightSize(const LayerType& layer)
{
  size_t weights = layer->Parameter().n_elem;

  if (layer->Model().size() > 0)
  {
    for (size_t i = 0; i < layer->Model().size(); ++i)
      weights += WeightSize(layer->Model()[i]);
  }

  return weights;
}

} // namespace ann
} // namespace mlpack

#endif
