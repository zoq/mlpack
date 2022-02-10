/**
 * @file methods/ann/ffn_impl.hpp
 * @author Marcus Edel
 *
 * Definition of the FFN class, which implements feed forward neural networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_FFN_IMPL_HPP
#define MLPACK_METHODS_ANN_FFN_IMPL_HPP

// In case it hasn't been included yet.
#include "ffn.hpp"

#include "make_alias.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::FFN(OutputLayerType outputLayer, InitializationRuleType initializeRule) :
    outputLayer(std::move(outputLayer)),
    initializeRule(std::move(initializeRule)),
    training(false),
    layerMemoryIsSet(false),
    inputDimensionsAreSet(false)
{
  /* Nothing to do here. */
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::FFN(const FFN& network):
    outputLayer(network.outputLayer),
    initializeRule(network.initializeRule),
    network(network.network),
    parameters(network.parameters),
    inputDimensions(network.inputDimensions),
    predictors(network.predictors),
    responses(network.responses),
    training(network.training),
    // These will be set correctly in the first Forward() call.
    layerMemoryIsSet(false),
    inputDimensionsAreSet(false)
{
  // Nothing to do.
};

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::FFN(FFN&& network):
    outputLayer(std::move(network.outputLayer)),
    initializeRule(std::move(network.initializeRule)),
    network(std::move(network.network)),
    parameters(std::move(network.parameters)),
    inputDimensions(std::move(network.inputDimensions)),
    predictors(std::move(network.predictors)),
    responses(std::move(network.responses)),
    training(std::move(network.training)),
    // Aliases will not be correct after a std::move(), so we will manually
    // reset them.
    layerMemoryIsSet(false),
    inputDimensionsAreSet(std::move(network.inputDimensionsAreSet))
{
  // Nothing to do.
};

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
FFN<OutputLayerType, InitializationRuleType, InputType, OutputType>& FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::operator=(FFN network)
{
  Swap(network);
  return *this;
};

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::ResetData(InputType predictors, InputType responses)
{
  this->predictors = std::move(predictors);
  this->responses = std::move(responses);

  // Set the network to training mode.
  SetNetworkMode(true);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename OptimizerType>
typename std::enable_if<
      HasMaxIterations<OptimizerType, size_t&(OptimizerType::*)()>::value,
      void>::type
FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::WarnMessageMaxIterations(OptimizerType& optimizer, size_t samples) const
{
  if (optimizer.MaxIterations() < samples &&
      optimizer.MaxIterations() != 0)
  {
    Log::Warn << "The optimizer's maximum number of iterations "
        << "is less than the size of the dataset; the "
        << "optimizer will not pass over the entire "
        << "dataset. To fix this, modify the maximum "
        << "number of iterations to be at least equal "
        << "to the number of points of your dataset "
        << "(" << samples << ")." << std::endl;
  }
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename OptimizerType>
typename std::enable_if<
      !HasMaxIterations<OptimizerType, size_t&(OptimizerType::*)()>
      ::value, void>::type
FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::WarnMessageMaxIterations(OptimizerType& /* optimizer */,
                            size_t /* samples */) const
{
  // Nothing to do here.
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename OptimizerType, typename... CallbackTypes>
double FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Train(InputType predictors,
         InputType responses,
         OptimizerType& optimizer,
         CallbackTypes&&... callbacks)
{
  ResetData(std::move(predictors), std::move(responses));

  WarnMessageMaxIterations<OptimizerType>(optimizer, this->predictors.n_cols);

  // Ensure that the network can be used.
  CheckNetwork("FFN::Train()", this->predictors.n_rows, true, true);

  // Train the model.
  Timer::Start("ffn_optimization");
  const double out = optimizer.Optimize(*this, parameters, callbacks...);
  Timer::Stop("ffn_optimization");

  Log::Info << "FFN::FFN(): final objective of trained model is " << out
      << "." << std::endl;
  return out;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename OptimizerType, typename... CallbackTypes>
double FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Train(InputType predictors,
         InputType responses,
         CallbackTypes&&... callbacks)
{
  ResetData(std::move(predictors), std::move(responses));

  OptimizerType optimizer;

  WarnMessageMaxIterations<OptimizerType>(optimizer, this->predictors.n_cols);

  // Ensure that the network can be used.
  CheckNetwork("FFN::Train()", this->predictors.n_rows, true, true);

  // Train the model.
  Timer::Start("ffn_optimization");
  const double out = optimizer.Optimize(*this, parameters, callbacks...);
  Timer::Stop("ffn_optimization");

  Log::Info << "FFN::FFN(): final objective of trained model is " << out
      << "." << std::endl;
  return out;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename PredictorsType, typename ResponsesType>
void FFN<OutputLayerType, InitializationRuleType, InputType, OutputType
>::Forward(const PredictorsType& inputs, ResponsesType& results)
{
  Forward(inputs, results, 0, network.Network().size() - 1);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename PredictorsType, typename ResponsesType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Forward(const PredictorsType& inputs,
           ResponsesType& results,
           const size_t begin,
           const size_t end)
{
  // Sanity checking...
  if (end < begin)
    return;

  // Ensure the network is valid.
  CheckNetwork("FFN::Forward()", inputs.n_rows);

  // We must always store a copy of the forward pass in `networkOutputs` in case
  // we do a backward pass.
  networkOutput.set_size(network.OutputSize(), inputs.n_cols);
  network.Forward(inputs, networkOutput, begin, end);

  // It's possible the user passed `networkOutputs` as `results`; in this case,
  // we don't need to create an alias.
  if (&results != &networkOutput)
    results = networkOutput;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename PredictorsType, typename TargetsType, typename GradientsType>
double FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Backward(const PredictorsType& inputs,
            const TargetsType& targets,
            GradientsType& gradients)
{
  const double res = outputLayer.Forward(networkOutput, targets) +
      network.Loss();

  // Compute the error of the output layer.
  outputLayer.Backward(networkOutput, targets, error);

  // Perform the backward pass.
  network.Backward(networkOutput, error, networkDelta);

  // Now compute the gradients.
  // The gradient should have the same size as the parameters.
  gradients.set_size(parameters.n_rows, parameters.n_cols);
  network.Gradient(inputs, error, gradients);

  return res;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Predict(InputType predictors, OutputType& results, const size_t batchSize)
{
  // Ensure that the network is configured correctly.
  CheckNetwork("FFN::Predict()", predictors.n_rows, true, false);

  results.set_size(network.OutputSize(), predictors.n_cols);

  for (size_t i = 0; i < predictors.n_cols; i += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize,
        size_t(predictors.n_cols) - i);

    InputType predictorAlias(predictors.colptr(i), predictors.n_rows,
        effectiveBatchSize, false, true);
    OutputType resultAlias(results.colptr(i), results.n_rows,
        effectiveBatchSize, false, true);

    Forward(predictorAlias, resultAlias);
  }
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename PredictorsType, typename ResponsesType>
double FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Evaluate(const PredictorsType& predictors, const ResponsesType& responses)
{
  // Sanity check: ensure network is valid.
  CheckNetwork("FFN::Evaluate()", predictors.n_rows);

  // Set networkOutput to the right size if needed, then perform the forward
  // pass.
  network.Forward(predictors, networkOutput);

  return outputLayer.Forward(networkOutput, responses) + network.Loss();
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
double FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Evaluate(const OutputType& parameters)
{
  double res = 0;
  for (size_t i = 0; i < predictors.n_cols; ++i)
    res += Evaluate(parameters, i, 1);

  return res;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
double FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Evaluate(const OutputType& /* parameters */,
            const size_t begin,
            const size_t batchSize)
{
  CheckNetwork("FFN::Evaluate()", predictors.n_rows);

  // Set networkOutput to the right size if needed, then perform the forward
  // pass.
  networkOutput.set_size(network.OutputSize(), batchSize);
  network.Forward(predictors.cols(begin, begin + batchSize - 1), networkOutput);

  return outputLayer.Forward(networkOutput,
      responses.cols(begin, begin + batchSize - 1)) + network.Loss();
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
double FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::EvaluateWithGradient(const OutputType& parameters, OutputType& gradient)
{
  double res = 0;
  for (size_t i = 0; i < predictors.n_cols; ++i)
    res += EvaluateWithGradient(parameters, i, gradient, 1);

  return res;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
double FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::EvaluateWithGradient(const OutputType& parameters,
                        const size_t begin,
                        OutputType& gradient,
                        const size_t batchSize)
{
  CheckNetwork("FFN::EvaluateWithGradient()", predictors.n_rows);

  // Set networkOutput to the right size if needed, then perform the forward
  // pass.
  networkOutput.set_size(network.OutputSize(), batchSize);

  std::cout << predictors.cols(begin, begin + 1 - 1).n_rows << std::endl;
  std::cout << predictors.cols(begin, begin + 1 - 1).n_cols << std::endl;
  std::cout << predictors.cols(begin, begin + 1 - 1) << std::endl;
  std::cout << "----------------------------------------------------\n";
  std::cout << predictors.cols(begin, begin + 2 - 1) << std::endl;
  exit(0);
  network.Forward(predictors.cols(begin, begin + batchSize - 1), networkOutput);

  const double obj = outputLayer.Forward(networkOutput,
      responses.cols(begin, begin + batchSize - 1)) + network.Loss();

  // Now perform the backward pass.
  outputLayer.Backward(networkOutput,
      responses.cols(begin, begin + batchSize - 1), error);

  // The delta should have the same size as the input.
  networkDelta.set_size(predictors.n_rows, batchSize);
  network.Backward(networkOutput, error, networkDelta);

  // Now compute the gradients.
  // The gradient should have the same size as the parameters.
  gradient.set_size(parameters.n_rows, parameters.n_cols);
  network.Gradient(predictors.cols(begin, begin + batchSize - 1), error,
      gradient);

  return obj;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Gradient(const OutputType& parameters,
            const size_t begin,
            OutputType& gradient,
            const size_t batchSize)
{
  this->EvaluateWithGradient(parameters, begin, gradient, batchSize);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Shuffle()
{
  math::ShuffleData(predictors, responses, predictors, responses);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Reset(const size_t inputDimensionality)
{
  parameters.clear();

  if (inputDimensionality != 0)
  {
    CheckNetwork("FFN::Reset()", inputDimensionality, true, false);
  }
  else
  {
    const size_t inputDims = std::accumulate(inputDimensions.begin(),
        inputDimensions.end(), 0);
    CheckNetwork("FFN::Reset()", inputDims, true, false);
  }
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::InitializeWeights()
{
  // Set the network to testing mode.
  SetNetworkMode(false);

  // Reset the network parameters with the given initialization rule.
  NetworkInitialization<InitializationRuleType> networkInit(initializeRule);
  networkInit.Initialize(network.Network(), parameters);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::SetLayerMemory()
{
  size_t totalWeightSize = network.WeightSize();

  Log::Assert(totalWeightSize == parameters.n_elem,
      "FFN::SetLayerMemory(): total layer weight size does not match parameter "
      "size!");

  network.SetWeights(parameters.memptr());
  layerMemoryIsSet = true;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::CheckNetwork(const std::string& functionName,
                const size_t inputDimensionality,
                const bool setMode,
                const bool training)
{
  // If the network is empty, we can't do anything.
  if (network.Network().size() == 0)
  {
    throw std::invalid_argument(functionName + ": cannot use network with no "
        "layers!");
  }

  // Next, check that the input dimensions for each layer are correct.  Note
  // that this will throw an exception if the user has passed data that does not
  // match this->inputDimensions.
  if (!inputDimensionsAreSet)
    UpdateDimensions(functionName, inputDimensionality);

  // We may need to initialize the `parameters` matrix if it is empty.
  if (parameters.is_empty())
    InitializeWeights();

  // Make sure each layer is pointing at the right memory.
  if (!layerMemoryIsSet)
    SetLayerMemory();

  // Finally, set the layers of the network to the right mode if the user
  // requested it.
  if (setMode)
    SetNetworkMode(training);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::UpdateDimensions(const std::string& functionName,
                    const size_t inputDimensionality)
{
  // If the input dimensions are completely unset, then assume our input is
  // flat.
  if (inputDimensions.size() == 0)
    inputDimensions = { inputDimensionality };

  const size_t totalInputSize = std::accumulate(inputDimensions.begin(),
      inputDimensions.end(), 0);

  // TODO: improve this error message...
  if (totalInputSize != inputDimensionality)
  {
    throw std::logic_error(functionName + ": input size does not match expected"
        " size set with InputDimensions()!");
  }

  // If the input dimensions have not changed from what has been computed
  // before, we can terminate early---the network already has its dimensions
  // set.
  if (inputDimensions == network.InputDimensions())
  {
    inputDimensionsAreSet = true;
    return;
  }

  network.InputDimensions() = std::move(inputDimensions);
  network.ComputeOutputDimensions();
  inputDimensionsAreSet = true;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::SetNetworkMode(const bool training)
{
  this->training = training;
  network.Training() = this->training;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
template<typename Archive>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::serialize(Archive& ar, const uint32_t /* version */)
{
  // Serialize the output layer and initialization rule.
  ar(CEREAL_NVP(outputLayer));
  ar(CEREAL_NVP(initializeRule));

  // Serialize the network itself.
  ar(CEREAL_NVP(network));
  ar(CEREAL_NVP(parameters));

  // Serialize the expected input size.
  ar(CEREAL_NVP(inputDimensions));
  ar(CEREAL_NVP(training));

  // If we are loading, we need to initialize the weights.
  if (cereal::is_loading<Archive>())
  {
    // We can clear these members, since it's not possible to serialize in the
    // middle of training and resume.
    predictors.clear();
    responses.clear();

    networkOutput.clear();
    networkDelta.clear();

    layerMemoryIsSet = false;
    inputDimensionsAreSet = false;

    // The weights in `parameters` will be correctly set for each layer in the
    // first call to Forward().
  }
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename InputType,
         typename OutputType>
void FFN<
    OutputLayerType,
    InitializationRuleType,
    InputType,
    OutputType
>::Swap(FFN& network)
{
  std::swap(outputLayer, network.outputLayer);
  std::swap(initializeRule, network.initializeRule);
  std::swap(this->network, network.network);
  std::swap(parameters, network.parameters);
  std::swap(inputDimensions, network.inputDimensions);
  std::swap(predictors, network.predictors);
  std::swap(responses, network.responses);
  std::swap(training, network.training);
  std::swap(inputDimensionsAreSet, network.inputDimensionsAreSet);
  std::swap(networkOutput, network.networkOutput);
  std::swap(networkDelta, network.networkDelta);

  // std::swap() will not preserve Armadillo aliases correctly, so we will reset
  // those.
  layerMemoryIsSet = false;
};

} // namespace ann
} // namespace mlpack

#endif
