/**
 * @file methods/ann/layer/recurrent_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Recurrent class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RECURRENT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_RECURRENT_IMPL_HPP

// In case it hasn't yet been included.
#include "recurrent.hpp"

#include "../util/deleter.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
RecurrentType<InputType, OutputType>::RecurrentType() :
    rho(0),
    forwardStep(0),
    backwardStep(0),
    gradientStep(0),
    deterministic(false),
    ownsLayer(false)
{
  // Nothing to do.
}

template<typename InputType, typename OutputType>
template<
    typename StartModuleType,
    typename InputModuleType,
    typename FeedbackModuleType,
    typename TransferModuleType
>
RecurrentType<InputType, OutputType>::RecurrentType(
    const StartModuleType& start,
    const InputModuleType& input,
    const FeedbackModuleType& feedback,
    const TransferModuleType& transfer,
    const size_t rho) :
    startModule(new StartModuleType(start)),
    inputModule(new InputModuleType(input)),
    feedbackModule(new FeedbackModuleType(feedback)),
    transferModule(new TransferModuleType(transfer)),
    rho(rho),
    forwardStep(0),
    backwardStep(0),
    gradientStep(0),
    deterministic(false),
    ownsLayer(true)
{
  initialModule = new Sequential();
  mergeModule = new AddMerge(false, false, false);
  recurrentModule = new Sequential(false, false);

  initialModule->Add(inputModule)
  initialModule->Add(startModule)
  initialModule->Add(transferModule)

  mergeModule->Add(inputModule);
  mergeModule->Add(feedbackModule)
  recurrentModule->Add(mergeModule)
  recurrentModule->Add(transferModule)

  network.push_back(initialModule);
  network.push_back(mergeModule);
  network.push_back(feedbackModule);
  network.push_back(recurrentModule);
}

template<typename InputType, typename OutputType>
RecurrentType<InputType, OutputType>::RecurrentType(
    const RecurrentType& network) :
    rho(network.rho),
    forwardStep(network.forwardStep),
    backwardStep(network.backwardStep),
    gradientStep(network.gradientStep),
    deterministic(network.deterministic),
    ownsLayer(network.ownsLayer)
{
  startModule = network.startModule->Clone();
  inputModule = network.inputModule->Clone();
  feedbackModule = network.feedbackModule->Clone();
  transferModule = network.transferModule->Clone();
  initialModule = new Sequential();
  mergeModule = new AddMerge(false, false, false);
  recurrentModule = new Sequential(false, false);

  initialModule->Add(inputModule)
  initialModule->Add(startModule)
  initialModule->Add(transferModule)

  mergeModule->Add(inputModule);
  mergeModule->Add(feedbackModule)
  recurrentModule->Add(mergeModule)
  recurrentModule->Add(transferModule)
  this->network.push_back(initialModule);
  this->network.push_back(mergeModule);
  this->network.push_back(feedbackModule);
  this->network.push_back(recurrentModule);
}

template<typename InputType, typename OutputType>
size_t RecurrentType<InputType, OutputType>::InputShape() const
{
  const size_t inputShapeStartModule = startModule->InputShape();
  // Return the input shape of the first module that we have.
  if (inputShapeStartModule != 0)
  {
    return inputShapeStartModule;
  }
  // If input shape of first module is 0.
  else
  {
    // Return input shape of the second module that we have.
    const size_t inputShapeInputModule =inputModule->InputShape();
    if (inputShapeInputModule != 0)
    {
      return inputShapeInputModule;
    // If the input shape of second module is 0.
    }
    else
    {
      // Return input shape of the third module that we have.
      const size_t inputShapeFeedbackModule = feedbackModule->InputShape();
      if (inputShapeFeedbackModule != 0)
      {
        return inputShapeFeedbackModule;
      // If the input shape of the third module is 0.
      }
      else
      {
        // Return the shape of the fourth module that we have.
        const size_t inputShapeTransferModule = transferModule->InputShape();
        if (inputShapeTransferModule != 0)
        {
          return inputShapeTransferModule;
        }
        // If the input shape of the fourth module is 0.
        else
          return 0;
      }
    }
  }
}

template<typename InputType, typename OutputType>
void RecurrentType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  if (forwardStep == 0)
  {
    initialModule->Forward(input, output);
  }
  else
  {
    inputModule->Forward(input, inputModule->OutputParameter());

    feedbackModule->Forward(transferModule->OutputParameter(),
        feedbackModule->OutputParameter());

    recurrentModule->Forward(input, output);
  }

  output = transferModule->OutputParameter();

  // Save the feedback output parameter when training the module.
  if (!deterministic)
  {
    feedbackOutputParameter.push_back(output);
  }

  forwardStep++;
  if (forwardStep == rho)
  {
    forwardStep = 0;
    backwardStep = 0;

    if (!recurrentError.is_empty())
    {
      recurrentError.zeros();
    }
  }
}

template<typename InputType, typename OutputType>
void RecurrentType<InputType, OutputType>::Backward(
    const InputType& /* input */,
    const InputType& gy,
    OutputType& g)
{
  if (!recurrentError.is_empty())
  {
    recurrentError += gy;
  }
  else
  {
    recurrentError = gy;
  }

  if (backwardStep < (rho - 1))
  {
    recurrentModule->Backward(recurrentModule->OutputParameter(),
        recurrentError,
        recurrentModule->Delta());
        

    inputModule->Backward(inputModule->OutputParameter(),
        recurrentModule->Delta(), g);
        

    feedbackModule->Backward(feedbackModule->OutputParameter(),
        recurrentModule->Delta(),
        feedbackModule->Delta());
  }
  else
  {
    initialModule->Backward(initialModule->OutputParameter(),
        recurrentError, g);
        
  }

  recurrentError = feedbackModule->Delta();
  backwardStep++;
}

template<typename InputType, typename OutputType>
void RecurrentType<InputType, OutputType>::Gradient(
    const InputType& input,
    const InputType& error,
    OutputType& /* gradient */)
{
  if (gradientStep < (rho - 1))
  {
    recurrentModule->Gradient(input, error);

    inputModule->Gradient(input, mergeModule->Delta());

    feedbackModule->Gradient(
        feedbackOutputParameter[feedbackOutputParameter.size() - 2 -
        gradientStep], mergeModule->Delta());
  }
  else
  {
    recurrentModule->Gradient().zero();
    inputModule->Gradient().zero();
    feedbackModule->Gradient().zero();

    initialModule->Gradient(input, startModule->Delta());
  }

  gradientStep++;
  if (gradientStep == rho)
  {
    gradientStep = 0;
    feedbackOutputParameter.clear();
  }
}

template<typename InputType, typename OutputType>
template<typename Archive>
void RecurrentType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  // Clean up memory, if we are loading.
  if (cereal::is_loading<Archive>())
  {
    // Clear old things, if needed.
    Deleter(recurrentModule);
    Deleter(initialModule);
    Deleter(startModule);
    network.clear();
  }

  ar(CEREAL_VARIANT_POINTER(startModule));
  ar(CEREAL_VARIANT_POINTER(inputModule));
  ar(CEREAL_VARIANT_POINTER(feedbackModule));
  ar(CEREAL_VARIANT_POINTER(transferModule));
  ar(CEREAL_NVP(rho));
  ar(CEREAL_NVP(ownsLayer));

  // Set up the network.
  if (cereal::is_loading<Archive>())
  {
    initialModule = new Sequential();
    mergeModule = new AddMerge(false, false, false);
    recurrentModule = new Sequential(false, false);

      initialModule->Add(inputModule)
      initialModule->Add(startModule)
      initialModule->Add(transferModule)

      mergeModule->Add(inputModule)
      mergeModule->Add(feedbackModule)
      recurrentModule->Add(mergeModule)
      recurrentModule->Add(transferModule)

    network.push_back(initialModule);
    network.push_back(mergeModule);
    network.push_back(feedbackModule);
    network.push_back(recurrentModule);
  }
}

} // namespace ann
} // namespace mlpack

#endif
