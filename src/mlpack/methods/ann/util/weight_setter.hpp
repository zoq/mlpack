/**
 * @file methods/ann/util/weight_setter.hpp
 * @author Aakash Kaushik
 *
 * Definition of the WeightSetter() function for different
 * layers and automatically directs any parameter to the right layer type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_UTIL_WEIGHT_SETTER_HPP
#define MLPACK_METHODS_ANN_UTIL_WEIGHT_SETTER_HPP

namespace mlpack {
namespace ann /** Artificial Neural Network. */{

/**
 * Execute the WeightSetter for the given layer.
 *
 * @tparam LayerType The type of the given layer.
 * @param layer The layer to execute the destructor for. 
 * @param weight The weight to assign to layers.
 * @param offset offset value.
 * @return modelOffset the offset of complete model. 
 */
template<typename LayerType, typename ParameterType>
size_t WeightSetter(const LayerType& layer,
    ParameterType& weight,
    const size_t offset = 0)
{
  //! Update the parameters if the module implements the Model() function.
  if (layer->Parameters().size() == 0 && layer->Model().size() > 0)
  {
    size_t modelOffset = 0;
    for (size_t i = 0; i < layer->Model().size(); ++i)
    {
      modelOffset += WeightSetter(layer->Model()[i], weight, 
          modelOffset + offset);
    }

    return modelOffset;
  }

  //! Update the parameters if the module implements the Parameters() function.
  else if (layer->Parameters().size() > 0 && layer->Model().size() == 0)
  {
    layer->Parameters() = arma::mat(weight.memptr() + offset,
      layer->Parameters().n_rows, layer->Parameters().n_cols, false, false);

    return layer->Parameters().n_elem;
  }

  else if (layer->Parameters().size() > 0 && layer->Model().size() > 0)
  {
    layer->Parameters() = arma::mat(weight.memptr() + offset,
      layer->Parameters().n_rows, layer->Parameters().n_cols, false, false);

    size_t modelOffset = layer->Parameters().n_elem;
    for (size_t i = 0; i < layer->Model().size(); ++i)
    {
      modelOffset += WeightSetter(layer->Model()[i], weight, 
          modelOffset + offset);
    }

    return modelOffset;
  }

  //! Do not update the parameters if the module doesn't implement the
  //! Parameters() or Model() function.
  return 0;

}

} // namespace ann
} // namespace mlpack

#endif
