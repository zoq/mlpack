/**
 * @file methods/ann/brnn_impl.hpp
 * @author Saksham Bansal
 *
 * Definition of the BRNN class, which implements bidirectional recurrent
 * neural networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_BRNN_IMPL_HPP
#define MLPACK_METHODS_ANN_BRNN_IMPL_HPP

// In case it hasn't been included yet.
#include "brnn.hpp"

#include "util/run_set.hpp"
#include "util/weight_size.hpp"
#include "util/load_output_parameter.hpp"
#include "util/save_output_parameter.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename InputType, typename OutputType>
BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, InputType, OutputType>::BRNN(
    const size_t rho,
    const bool single,
    OutputLayerType outputLayer,
    MergeLayerType* mergeLayer,
    MergeOutputType* mergeOutput,
    InitializationRuleType initializeRule) :
    rho(rho),
    outputLayer(std::move(outputLayer)),
    mergeLayer(mergeLayer),
    mergeOutput(mergeOutput),
    initializeRule(std::move(initializeRule)),
    inputSize(0),
    outputSize(0),
    targetSize(0),
    reset(false),
    single(single),
    numFunctions(0),
    deterministic(true),
    forwardRNN(rho, single, outputLayer, initializeRule),
    backwardRNN(rho, single, outputLayer, initializeRule)
{
  /* Nothing to do here. */
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename InputType, typename OutputType>
BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, InputType, OutputType>::~BRNN()
{
  // Remove the last layers from the forward and backward RNNs, as they are held
  // in mergeLayer.  So, when we use delete with mergeLayer, those two
  // layers will be properly (and not doubly) freed.
  forwardRNN.network.pop_back();
  backwardRNN.network.pop_back();

  // Clean up layers that we allocated.
  delete mergeLayer;
  delete mergeOutput;
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename InputType, typename OutputType>
template<typename OptimizerType>
typename std::enable_if<
      HasMaxIterations<OptimizerType, size_t&(OptimizerType::*)()>
      ::value, void>::type
BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, InputType, OutputType>::WarnMessageMaxIterations
(OptimizerType& optimizer, size_t samples) const
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

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename InputType, typename OutputType>
template<typename OptimizerType>
typename std::enable_if<
      !HasMaxIterations<OptimizerType, size_t&(OptimizerType::*)()>
      ::value, void>::type
BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, InputType, OutputType>::WarnMessageMaxIterations
(OptimizerType& /* optimizer */, size_t /* samples */) const
{
  return;
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename InputType, typename OutputType>
template<typename OptimizerType>
double BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, InputType, OutputType>::Train(
    arma::cube predictors,
    arma::cube responses,
    OptimizerType& optimizer)
{
  numFunctions = responses.n_cols;

  this->predictors = std::move(predictors);
  this->responses = std::move(responses);

  this->deterministic = true;
  ResetDeterministic();

  if (!reset)
  {
    ResetParameters();
  }

  WarnMessageMaxIterations<OptimizerType>(optimizer, this->predictors.n_cols);

  // Train the model.
  Timer::Start("BRNN_optimization");
  const double out = optimizer.Optimize(*this, parameter);
  Timer::Stop("BRNN_optimization");

  Log::Info << "BRNN::BRNN(): final objective of trained model is " << out
      << "." << std::endl;
  return out;
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename InputType, typename OutputType>
template<typename OptimizerType>
double BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, InputType, OutputType>::Train(
    arma::cube predictors,
    arma::cube responses)
{
  numFunctions = responses.n_cols;

  this->predictors = std::move(predictors);
  this->responses = std::move(responses);

  this->deterministic = true;
  ResetDeterministic();

  if (!reset)
  {
    ResetParameters();
  }

  OptimizerType optimizer;

  WarnMessageMaxIterations<OptimizerType>(optimizer, this->predictors.n_cols);

  // Train the model.
  const double out = optimizer.Optimize(*this, parameter);

  Log::Info << "BRNN::BRNN(): final objective of trained model is " << out
      << "." << std::endl;
  return out;
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename InputType, typename OutputType>
void BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, InputType, OutputType>::Predict(
    arma::cube predictors, arma::cube& results, const size_t batchSize)
{
  forwardRNN.rho = backwardRNN.rho = rho;

  forwardRNN.ResetCells();
  backwardRNN.ResetCells();

  if (!deterministic)
  {
    deterministic = true;
    ResetDeterministic();
  }
  if (parameter.is_empty())
  {
    ResetParameters();
  }

  if (std::is_same<MergeLayerType, Concat<>>::value)
  {
    results = arma::zeros<arma::cube>(outputSize * 2, predictors.n_cols, rho);
  }
  else
  {
    results = arma::zeros<arma::cube>(outputSize, predictors.n_cols, rho);
  }

  std::vector<arma::mat> results1, results2;
  arma::mat input;

  // Forward both RNN's from opposite directions.
  for (size_t begin = 0; begin < predictors.n_cols; begin += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize,
        size_t(predictors.n_cols - begin));
    for (size_t seqNum = 0; seqNum < rho; ++seqNum)
    {
      forwardRNN.Forward(arma::mat(
          predictors.slice(seqNum).colptr(begin),
          predictors.n_rows, effectiveBatchSize, false, true));
      backwardRNN.Forward(std::move(arma::mat(
          predictors.slice(rho - seqNum - 1).colptr(begin),
          predictors.n_rows, effectiveBatchSize, false, true)));

      SaveOutputParameter(forwardRNN.network.back(), results1)
      SaveOutputParameter(backwardRNN.network.back(), results2),
    }
    reverse(results1.begin(), results1.end());

    // Forward outputs from both RNN's through merge layer for each time step.
    for (size_t seqNum = 0; seqNum < rho; ++seqNum)
    {
      LoadOutputParameter(forwardRNN.network.back(), results1);
      LoadOutputParameter(backwardRNN.network.back(), results2);          

      mergeLayer->Forward(input,
          mergeLayer->OutputParameter());
      mergeOutput->Forward(
          mergeLayer->OutputParameter(),
          mergeOutput->OutputParameter());
      results.slice(seqNum).submat(0, begin, results.n_rows - 1, begin +
          effectiveBatchSize - 1) =
          mergeOutput->OutputParameter();
    }
  }
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename InputType, typename OutputType>
double BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, InputType, OutputType>::Evaluate(
    const arma::mat& /* parameters */,
    const size_t begin,
    const size_t batchSize,
    const bool deterministic)
{
  forwardRNN.rho = backwardRNN.rho = rho;
  if (parameter.is_empty())
  {
    ResetParameters();
  }

  if (deterministic != this->deterministic)
  {
    this->deterministic = deterministic;
    ResetDeterministic();
  }

  if (!inputSize)
  {
    inputSize = predictors.n_rows;
    targetSize = responses.n_rows;
  }
  else if (targetSize == 0)
  {
    targetSize = responses.n_rows;
  }

  forwardRNN.ResetCells();
  backwardRNN.ResetCells();

  double performance = 0;
  size_t responseSeq = 0;

  std::vector<arma::mat> results1, results2;
  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    forwardRNN.Forward(arma::mat(
        predictors.slice(seqNum).colptr(begin),
        predictors.n_rows, batchSize, false, true));
    backwardRNN.Forward(arma::mat(
        predictors.slice(rho - seqNum - 1).colptr(begin),
        predictors.n_rows, batchSize, false, true));

    SaveOutputParameter(forwardRNN.network.back(), results1);
    SaveOutputParameter(backwardRNN.network.back(), results2);
  }
  if (outputSize == 0)
  {
    outputSize = (forwardRNN.network.back()->OutputParameter()).n_elem /
        batchSize;
    forwardRNN.outputSize = backwardRNN.outputSize = outputSize;
  }
  reverse(results1.begin(), results1.end());

  // Performance calculation after forwarding through merge layer.
  arma::mat input;
  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    if (!single)
    {
      responseSeq = seqNum;
    }
    LoadOutputParameter(forwardRNN.network.back(), results1);
    LoadOutputParameter(backwardRNN.network.back(), results2);

    mergeLayer->Forward(input,
        mergeLayer->OutputParameter());
    mergeOutput->Forward(
        mergeLayer->OutputParameter(),
        mergeOutput->OutputParameter());
    performance += outputLayer.Forward(
        mergeOutput->OutputParameter(),
        arma::mat(responses.slice(responseSeq).colptr(begin),
        responses.n_rows, batchSize, false, true));
  }
  return performance;
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename InputType, typename OutputType>
double BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, InputType, OutputType>::Evaluate(
    const arma::mat& parameters,
    const size_t begin,
    const size_t batchSize)
{
  return Evaluate(parameters, begin, batchSize, true);
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename InputType, typename OutputType>
template<typename GradType>
double BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, InputType, OutputType>::
EvaluateWithGradient(const arma::mat& /* parameters */,
                     const size_t begin,
                     GradType& gradient,
                     const size_t batchSize)
{
  forwardRNN.rho = backwardRNN.rho = rho;
  if (gradient.is_empty())
  {
    if (parameter.is_empty())
    {
      ResetParameters();
    }
    gradient = arma::zeros<arma::mat>(parameter.n_rows, parameter.n_cols);
  }
  else
  {
    gradient.zeros();
  }

  if (backwardGradient.is_empty())
  {
    backwardGradient = arma::zeros<arma::mat>(
        parameter.n_rows/ 2,
        parameter.n_cols);
    forwardGradient = arma::zeros<arma::mat>(
        parameter.n_rows/ 2,
        parameter.n_cols);
  }
  if (this->deterministic)
  {
    this->deterministic = false;
    ResetDeterministic();
  }

  if (!inputSize)
  {
    inputSize = predictors.n_rows;
    targetSize = responses.n_rows;
  }
  else if (targetSize == 0)
  {
    targetSize = responses.n_rows;
  }

  forwardRNN.ResetCells();
  backwardRNN.ResetCells();
  size_t networkSize = backwardRNN.network.size();

  // Forward propogation from both directions.
  std::vector<arma::mat> results1, results2;
  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    forwardRNN.Forward(arma::mat(
        predictors.slice(seqNum).colptr(begin),
        predictors.n_rows, batchSize, false, true));
    backwardRNN.Forward(arma::mat(
        predictors.slice(rho - seqNum - 1).colptr(begin),
        predictors.n_rows, batchSize, false, true));

    for (size_t l = 0; l < networkSize; ++l)
    {
      SaveOutputParameter(forwardRNN.network[l], forwardRNNOutputParameter);
      SaveOutputParameter(backwardRNN.network[l], backwardRNNOutputParameter);
    }
    SaveOutputParameter(forwardRNN.network.back(), results1);
    SaveOutputParameter(backwardRNN.network.back(), results2);
  }
  if (outputSize == 0)
  {
    outputSize = (forwardRNN.network.back()->OutputParameter()).n_elem /
        batchSize;
    forwardRNN.outputSize = backwardRNN.outputSize = outputSize;
  }

  arma::cube results;
  if (std::is_same<MergeLayerType, Concat<>>::value)
  {
    results = arma::zeros<arma::cube>(outputSize * 2, batchSize, rho);
  }
  else
  {
    results = arma::zeros<arma::cube>(outputSize, batchSize, rho);
  }

  double performance = 0;
  size_t responseSeq = 0;
  arma::mat input;

  reverse(results1.begin(), results1.end());
  // Performance calculation here.
  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    if (!single)
    {
      responseSeq = seqNum;
    }
    LoadOutputParameter(forwardRNN.network.back(), results1);
    LoadOutputParameter(backwardRNN.network.back(), results2);
    mergeLayer->Forward(input,
        mergeLayer->OutputParameter());
    mergeOutput->Forward(
        mergeLayer->OutputParameter(),
        results.slice(seqNum));
    performance += outputLayer.Forward(results.slice(seqNum),
        arma::mat(responses.slice(responseSeq).colptr(begin),
        responses.n_rows, batchSize, false, true));
  }

  // Calculate and storing delta parameters from output for t = 1 to T.
  arma::mat delta;
  std::vector<arma::mat> allDelta;

  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    if (single && seqNum > 0)
    {
      error.zeros();
    }
    else if (single && seqNum == 0)
    {
      outputLayer.Backward(results.slice(seqNum),
          arma::mat(responses.slice(0).colptr(begin),
          responses.n_rows, batchSize, false, true), error);
    }
    else
    {
      outputLayer.Backward(results.slice(seqNum),
          arma::mat(responses.slice(seqNum).colptr(begin),
          responses.n_rows, batchSize, false, true), error);
    }

    mergeOutput->Backward(results.slice(seqNum), error, delta);
    allDelta.push_back(arma::mat(delta));
  }

  // BPTT ForwardRNN from t = T to 1.
  totalGradient = arma::mat(gradient.memptr(),
      parameter.n_elem / 2, 1, false, false);

  forwardGradient.zeros();
  forwardRNN.ResetGradients(forwardGradient);
  backwardGradient.zeros();
  backwardRNN.ResetGradients(backwardGradient);

  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    forwardGradient.zeros();
    for (size_t l = 0; l < networkSize; ++l)
    {
      LoadOutputParameter(forwardRNN.network[networkSize - 1 - l],
          forwardRNNOutputParameter)
    }
    mergeLayer->Backward(
        forwardRNN.network.back()->OutputParameter(),
        allDelta[rho - seqNum - 1], delta, 0);

    for (size_t i = 2; i < networkSize; ++i)
    {
      forwardRNN.network[networkSize - i]->Backward(
          forwardRNN.network[networkSize - i]->OutputParameter(),
          forwardRNN.network[networkSize - i + 1]->Delta(),
          forwardRNN.network[networkSize - i]->Delta());
    }
    forwardRNN.Gradient(
        arma::mat(predictors.slice(rho - seqNum - 1).colptr(begin),
        predictors.n_rows, batchSize, false, true));
    mergeLayer->Gradient(
        forwardRNN.network[networkSize - 2]->OutputParameter(),
        allDelta[rho - seqNum - 1], 0);
    totalGradient += forwardGradient;
  }

  // BPTT BackwardRNN from t = 1 to T.
  totalGradient = arma::mat(gradient.memptr() + parameter.n_elem/2,
      parameter.n_elem/2, 1, false, false);

  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    backwardGradient.zeros();
    for (size_t l = 0; l < networkSize; ++l)
    {
      LoadOutputParameter(backwardRNN.network[networkSize - 1 - l],
          backwardRNNOutputParameter)
    }
    mergeLayer->Backward(
        backwardRNN.network.back()->OutputParameter(),
        allDelta[seqNum], delta, 1);
    for (size_t i = 2; i < networkSize; ++i)
    {
      backwardRNN.network[networkSize - i]->Backward(
        backwardRNN.network[networkSize - i]->OutputParameter(),
        backwardRNN.network[networkSize - i + 1]->Delta(),
        backwardRNN.network[networkSize - i]->Delta());
        
    }

    backwardRNN.Gradient(
        arma::mat(predictors.slice(seqNum).colptr(begin),
        predictors.n_rows, batchSize, false, true));
    mergeLayer->Gradient(std::move(
        backwardRNN.network[networkSize - 2]->OutputParameter()),
        allDelta[seqNum], 1);
    totalGradient += backwardGradient;
  }
  return performance;
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename InputType, typename OutputType>
void BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, InputType, OutputType>::Gradient(
    const arma::mat& parameters,
    const size_t begin,
    arma::mat& gradient,
    const size_t batchSize)
{
  this->EvaluateWithGradient(parameters, begin, gradient, batchSize);
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename InputType, typename OutputType>
void BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, InputType, OutputType>::Shuffle()
{
  arma::cube newPredictors, newResponses;
  math::ShuffleData(predictors, responses, newPredictors, newResponses);

  predictors = std::move(newPredictors);
  responses = std::move(newResponses);
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename InputType, typename OutputType>
template <class LayerType, class... Args>
void BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, InputType, OutputType>::Add(Args... args)
{
  forwardRNN.network.push_back(new LayerType(args...));
  backwardRNN.network.push_back(new LayerType(args...));
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename InputType, typename OutputType>
void BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, InputType, OutputType>::
Add(LayerTypes<InputType, OutputType>* layer)
{
  forwardRNN.network.push_back(layer);
  backwardRNN.network.push_back(layer->Clone());
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename InputType, typename OutputType>
void BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, InputType, OutputType>::ResetParameters()
{
  if (!reset)
  {
    // TODO: what if we call ResetParameters() multiple times?  Do we have to
    // remove any existing mergeLayer?
    mergeLayer->Add(forwardRNN.network.back());
    mergeLayer->Add(backwardRNN.network.back());
    RunSet(mergeLayer, false);
  }

  ResetDeterministic();

  // Reset the network parameter with the given initialization rule.
  NetworkInitialization<InitializationRuleType> networkInit(initializeRule);
  
  size_t rnnWeights = 0;
  for (size_t i = 0; i < forwardRNN.network.size(); ++i)
  {
    rnnWeights += WeightSize(forwardRNN.network[i]);
  }

  parameter.set_size(2 * rnnWeights, 1);

  forwardRNN.Parameters() = arma::mat(parameter.memptr(),
      rnnWeights, 1, false, false);
  backwardRNN.Parameters() = arma::mat(parameter.memptr() + rnnWeights,
      rnnWeights, 1, false, false);

  // Initialize the forward RNN parameters
  networkInit.Initialize(forwardRNN.network, parameter);

  // Initialize the backward RNN parameters
  networkInit.Initialize(backwardRNN.network, parameter, rnnWeights);

  reset = forwardRNN.reset = backwardRNN.reset = true;
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename InputType, typename OutputType>
void BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, InputType, OutputType>::Reset()
{
  ResetParameters();
  forwardRNN.ResetCells();
  backwardRNN.ResetCells();
  forwardGradient.zeros();
  forwardRNN.ResetGradients(forwardGradient);
  backwardGradient.zeros();
  backwardRNN.ResetGradients(backwardGradient);
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename InputType, typename OutputType>
void BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, InputType, OutputType>::ResetDeterministic()
{
  forwardRNN.deterministic = this->deterministic;
  backwardRNN.deterministic = this->deterministic;
  forwardRNN.ResetDeterministic();
  backwardRNN.ResetDeterministic();
}

template<typename OutputLayerType, typename MergeLayerType,
         typename MergeOutputType, typename InitializationRuleType,
         typename InputType, typename OutputType>
template<typename Archive>
void BRNN<OutputLayerType, MergeLayerType, MergeOutputType,
    InitializationRuleType, InputType, OutputType>::serialize(
    Archive& ar, const uint32_t version)
{
  ar(CEREAL_NVP(parameter));
  ar(CEREAL_NVP(backwardRNN));
  ar(CEREAL_NVP(forwardRNN));

  // TODO: are there more parameters to be serialized?
}

} // namespace ann
} // namespace mlpack

#endif
