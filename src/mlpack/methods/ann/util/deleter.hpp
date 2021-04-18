/**
 * @file methods/ann/util/deleter.hpp
 * @author Aakash Kaushik
 *
 * Definition of the Deleter() function which executes the destructor of 
 * the given layer or network.
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
 * Execute the destructor of the given layer.
 *
 * @tparam LayerType The type of the given layer.
 * @param layer The layer to execute the destructor for. 
 */
template<typename LayerType>
void Deleter(const LayerType layer)
{
  delete layer;

  if (layer->Model().size() > 0)
  {
    for (size_t i = 0; i < layer->Model().size(); ++i)
      Deleter(layer->Model()[i]);
  }
}

} // namespace ann
} // namespace mlpack

#endif
