/**
 * @file methods/ann/layer/recurrent_attention_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the RecurrentAttention class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RECURRENT_ATTENTION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_RECURRENT_ATTENTION_IMPL_HPP

// In case it hasn't yet been included.
#include "recurrent_attention.hpp"

#include "../util/weight_size.hpp"
#include "../util/gradient_set.hpp"
#include "../util/reset_update.hpp"
#include "../util/gradient_update.hpp"
#include "../util/load_output_parameter.hpp"
#include "../util/save_output_parameter.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
RecurrentAttentionType<InputType, OutputType>::RecurrentAttentionType() :
    rho(0),
    forwardStep(0),
    backwardStep(0),
    deterministic(false),
    outSize(0)
{
  // Nothing to do.
}

template <typename InputType, typename OutputType>
template<typename RNNModuleType, typename ActionModuleType>
RecurrentAttentionType<InputType, OutputType>::RecurrentAttentionType(
    const size_t outSize,
    const RNNModuleType& rnn,
    const ActionModuleType& action,
    const size_t rho) :
    outSize(outSize),
    rnnModule(new RNNModuleType(rnn)),
    actionModule(new ActionModuleType(action)),
    rho(rho),
    forwardStep(0),
    backwardStep(0),
    deterministic(false)
{
  network.push_back(rnnModule);
  network.push_back(actionModule);
}

template<typename InputType, typename OutputType>
void RecurrentAttentionType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  // Initialize the action input.
  if (initialInput.is_empty())
  {
    initialInput = arma::zeros(outSize, input.n_cols);
  }

  // Propagate through the action and recurrent module.
  for (forwardStep = 0; forwardStep < rho; ++forwardStep)
  {
    if (forwardStep == 0)
    {
      actionModule->Forward(initialInput, actionModule->OutputParameter());
    }
    else
    {
      actionModule->Forward(rnnModule->OutputParameter(),
          actionModule->OutputParameter());
    }

    // Initialize the glimpse input.
    arma::mat glimpseInput = arma::zeros(input.n_elem, 2);
    glimpseInput.col(0) = input;
    glimpseInput.submat(0, 1, actionModule->OutputParameter().n_elem - 1, 1) =
        actionModule->OutputParameter();

    rnnModule->Forward(glimpseInput, rnnModule->OutputParameter());

    // Save the output parameter when training the module.
    if (!deterministic)
    {
      for (size_t l = 0; l < network.size(); ++l)
      {
        SaveOutputParameter(network[l], moduleOutputParameter);
      }
    }
  }

  output = rnnModule->OutputParameter();

  forwardStep = 0;
  backwardStep = 0;
}

template<typename InputType, typename OutputType>
void RecurrentAttentionType<InputType, OutputType>::Backward(
    const InputType& /* input */,
    const InputType& gy,
    OutputType& g)
{
  if (intermediateGradient.is_empty() && backwardStep == 0)
  {
    // Initialize the attention gradients.
    size_t weights = WeightSize(rnnModule) +
        WeightSize(actionModule);

    intermediateGradient = arma::zeros(weights, 1);
    attentionGradient = arma::zeros(weights, 1);

    // Initialize the action error.
    actionError = arma::zeros(
      actionModule->OutputParameter().n_rows,
      actionModule->OutputParameter().n_cols);
  }

  // Propagate the attention gradients.
  if (backwardStep == 0)
  {
    size_t offset = 0;
    offset += GradientSet(rnnModule, intermediateGradient,
        offset);
    GradientSet(actionModule, intermediateGradient, offset);

    attentionGradient.zeros();
  }

  // Back-propagate through time.
  for (; backwardStep < rho; backwardStep++)
  {
    if (backwardStep == 0)
    {
      recurrentError = gy;
    }
    else
    {
      recurrentError = actionDelta;
    }

    for (size_t l = 0; l < network.size(); ++l)
    {
      LoadOutputParameter(network[network.size() - 1 - l],
          moduleOutputParameter);
    }

    if (backwardStep == (rho - 1))
    {
      actionModule->Backward(actionModule->OutputParameter(),
          actionError, actionDelta);
    }
    else
    {
      actionModule->Backward(initialInput, actionError,
          actionDelta);
    }

    rnnModule->Backward(rnnModule->OutputParameter(),
        recurrentError, rnnDelta);

    if (backwardStep == 0)
    {
      g = rnnDelta.col(1);
    }
    else
    {
      g += rnnDelta.col(1);
    }

    IntermediateGradient();
  }
}

template<typename InputType, typename OutputType>
void RecurrentAttentionType<InputType, OutputType>::Gradient(
    const InputType& /* input */,
    const InputType& /* error */,
    OutputType& /* gradient */)
{
  size_t offset = 0;
  offset += GradientUpdate(rnnModule, attentionGradient, offset);
  GradientUpdate(actionModule, attentionGradient, offset);
}

template<typename InputType, typename OutputType>
template<typename Archive>
void RecurrentAttentionType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(rho));
  ar(CEREAL_NVP(outSize));
  ar(CEREAL_NVP(forwardStep));
  ar(CEREAL_NVP(backwardStep));

  ar(CEREAL_VARIANT_POINTER(rnnModule));
  ar(CEREAL_VARIANT_POINTER(actionModule));
}

} // namespace ann
} // namespace mlpack

#endif
