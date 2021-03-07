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


namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
RecurrentAttentionType<InputType, OutputType>::RecurrentAttentionType() :
    rho(0),
    forwardStep(0),
    backwardStep(0),
    deterministic(false)
{
  // Nothing to do.
}

template <typename InputType, typename OutputType>
template<typename RNNModuleType, typename ActionModuleType>
RecurrentAttentionType<InputType, OutputType>::RecurrentAttentionType(
    const size_t outSize,
    const RNNModuleType& rnn,
    const ActionModuleType& action,
    const size_t rho):
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
      actionModule->initialInput, actionModule->outputParameter());
    }
    else
    {

     actionModule->Forward(rnnModule->outputParameter(), actionModule->outputParameter());
    }

    // Initialize the glimpse input.
    arma::mat glimpseInput = arma::zeros(input.n_elem, 2);
    glimpseInput.col(0) = input;
    glimpseInput.submat(0, 1, actionModule->outputParameter()).n_elem -1, 1) = actionModule->outputParameter();

    rnnModule->Forward(glimpseInput, rnnModule->outputParameter());

    // Save the output parameter when training the module.
    if (!deterministic)
    {
      for (size_t l = 0; l < network.size(); ++l)
      {
            network[l]->SaveOutputParameter(moduleOutputParameter);

      }
    }
  }

  output = rnnModule->outputParameter();

  forwardStep = 0;
  backwardStep = 0;
}

template<typename InputType, typename OutputType>

void RecurrentAttention<InputType, OutputType>::Backward(
    const InputType& /* input */,
    const OutputType& gy,
    OutputType& g)
{
  if (intermediateGradient.is_empty() && backwardStep == 0)
  {
    // Initialize the attention gradients.

    size_t weights = rnnModule->weightSize() + actionModule->weightSize();

    intermediateGradient = arma::zeros(weights, 1);
    attentionGradient = arma::zeros(weights, 1);

    // Initialize the action error.
      actionError = arma::zeros(actionModule->outputParameter().n_rows, actionModule->outputParameter().n_cols);
  }

  // Propagate the attention gradients.
  if (backwardStep == 0)
  {
    size_t offset = 0;
    offset += rnnmodule->GradientSet(offset->intermediateGradient());
    offset += actionModule->GradientSet(offset->intermediateGradient());

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
    network[network.size() - 1 - l]->LoadOutputParameter(moduleOutputParameter);

    }

    if (backwardStep == (rho - 1))
    {
      actionModule->Backward(actionModule->outputParameter(), actionError, actionDelta);
    }
    else
    {
      actionModule->Backward(initialInput, actionError, actionDelta);
    }

    rnnModule->Backward(rnnModule->outputParameter(), recurrentError, rnnDelta);

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
    const OutputType& /* error */,
    OutputType& /* gradient */)
 {
  size_t offset = 0;
  offset += rnnmodule->GradientUpdate(offset->attentionGradient());
  actionModule->GradientUpdate(offset->attentionGradient());
 }

template<typename InputType, typename OutputType>
template<typename Archive>
void RecurrentAttentionType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */);
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
