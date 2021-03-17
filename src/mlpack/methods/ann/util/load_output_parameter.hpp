/**
 * @file methods/ann/util/load_output_parameter.hpp
 * @author Aakash Kaushik
 *
 * Definition of the LoadOutputParameter which restores the output parameter using the given
 * parameter set.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_UTIL_LOAD_OUTPUT_PARAMETER_HPP
#define MLPACK_METHODS_ANN_UTIL_LOAD_OUTPUT_PARAMETER_HPP

namespace mlpack {
namespace ann /** Artificial Neural Network. */{

/**
 * Updates the layer parameters through a given parameter vector. 
 * 
 * @tparam LayerType The type of the given layer.
 * @tparam ParamVector The type of given vector, generally std::vector<arma::mat>.
 * @param layer The layer to update the paramter for. 
 * @param parameter The vector from which the parameters are updated. 
 */
template<typename LayerType, typename ParamVector>
void LoadOutputParameter(LayerType& layer, ParamVector& parameter)
{
  layer->OutputParameter() = parameter.back();
  parameter.pop_back();

  if (layer->Model() > 0)
  {
    for (size_t i = 0; i != layer.Model().size(); ++i)
      LoadOutputParameter(layer->Model()[i], parameter);
  }
}

} // namespace ann
} // namespace mlpack 

#endif