/**
 * @file methods/ann/util/gradient_set.hpp
 * @author Aakash Kaushik
 *
 * Definition of the GradientSet() function which updates the gradient 
 * parameter give the gradient set.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_UTIL_GRADIENT_SET_HPP
#define MLPACK_METHODS_ANN_UTIL_GRADIENT_SET_HPP

namespace mlpack {
namespace ann /** Artificial Neural Network. */{

/**
 * Update the gradient parameter give the gradient set.
 *
 * @tparam LayerType The type of the layer.
 * @tparam MatType The type of the gradient matrix 
 * @param layer The layer that the gradient is assigned to.
 * @param gradient gradient set to update gradient parameter. 
 * @param offset The beginning of the gradient portion we assign to the layer.
 * @return the layer offset.
 */
template<typename LayerType, typename MatType>
size_t GradientSet(
    const LayerType& layer, MatType& gradient, size_t offset = 0)
{
  layer->Gradient() = MatType(gradient.memptr() + offset,
      layer->Parameters().n_rows, layer->Parameters().n_cols, false, false);

  if (layer->Model().size() > 0)
  {
    size_t modelOffset = 0;

    for (size_t i = 0; i < layer->Model().size(); ++i)
    {
      modelOffset += GradientSet(layer->Model()[i], gradient,
          modelOffset + offset);
    }

  return modelOffset;
  }

  return layer->Parameters().n_elem;
}

} // namespace ann
} // namespace mlpack

#endif
