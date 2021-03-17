/**
 * @file methods/ann/util/save_output_parameter.hpp
 * @author Aakash Kaushik
 *
 * Definition of the SaveOutputParameter which saves the output parameter of
 * the given layer to the parameter set.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_UTIL_SAVE_OUTPUT_PARAMETER_HPP
#define MLPACK_METHODS_ANN_UTIL_SAVE_OUTPUT_PARAMETER_HPP

namespace mlpack {
namespace ann /** Artificial Neural Network. */{

/**
 * Saves the layer parameters to a given parameter vector. 
 * 
 * @tparam LayerType The type of the given layer.
 * @tparam ParamVector The type of given vector, generally std::vector<arma::mat>.
 * @param layer The layer to save the paramter from. 
 * @param parameter The vector in which the parameters are saved. 
 */
template<typename LayerType, typename ParamVector>
void SaveOutputParameter(LayerType& layer, ParamVector& parameter)
{
  parameter.push_back(layer->OutputParameter());

  if (layer->Model() > 0)
  {
    for (size_t i = 0; i != layer.Model().size(); ++i)
      SaveOutputParameter(layer->Model()[i], parameter);
  }
}

} // namespace ann
} // namespace mlpack 

#endif