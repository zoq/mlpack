/**
 * @file methods/ann/layer/gru_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of the GRU class, which implements a gru network
 * layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_GRU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_GRU_IMPL_HPP

// In case it hasn't yet been included.
#include "gru.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
GRUType<InputType, OutputType>::GRUType()
{
  // Nothing to do here.
}

template <typename InputType, typename OutputType>
GRUType<InputType, OutputType>::GRUType(
    const size_t inSize,
    const size_t outSize,
    const size_t rho) :
    inSize(inSize),
    outSize(outSize),
    rho(rho),
    batchSize(1),
    forwardStep(0),
    backwardStep(0),
    gradientStep(0),
    deterministic(false)
{
  // Input specific linear layers(for zt, rt, ot).
  input2GateModule = new Linear(inSize, 3 * outSize);

  // Previous output gates (for zt and rt).
  output2GateModule = new LinearNoBias(outSize, 2 * outSize);

  // Previous output gate for ot.
  outputHidden2GateModule = new LinearNoBias(outSize, outSize);

  network.push_back(input2GateModule);
  network.push_back(output2GateModule);
  network.push_back(outputHidden2GateModule);

  inputGateModule = new SigmoidLayer();
  forgetGateModule = new SigmoidLayer();
  hiddenStateModule = new TanHLayer();

  network.push_back(inputGateModule);
  network.push_back(hiddenStateModule);
  network.push_back(forgetGateModule);

  prevError = arma::zeros<InputType>(3 * outSize, batchSize);

  allZeros = arma::zeros<OutputType>(outSize, batchSize);

  outParameter.emplace_back(allZeros.memptr(),
      allZeros.n_rows, allZeros.n_cols, false, true);

  prevOutput = outParameter.begin();
  backIterator = outParameter.end();
  gradIterator = outParameter.end();
}

template<typename InputType, typename OutputType>
void GRUType<InputType, OutputType>::Forward(
    const InputType& input,
    OutputType& output)
{
  if (input.n_cols != batchSize)
  {
    batchSize = input.n_cols;
    prevError.resize(3 * outSize, batchSize);
    allZeros.zeros(outSize, batchSize);
    // Batch size better not change during an iteration...
    if (outParameter.size() > 1)
    {
      Log::Fatal << "GRU::Forward(): batch size cannot change during a "
          << "forward pass!" << std::endl;
    }

    outParameter.clear();
    outParameter.emplace_back(allZeros.memptr(),
        allZeros.n_rows, allZeros.n_cols, false, true);

    prevOutput = outParameter.begin();
    backIterator = outParameter.end();
    gradIterator = outParameter.end();
  }

  // Process the input linearly(zt, rt, ot).
  input2GateModule->Forward(input, input2GateModule->OutputParameter());

  // Process the output(zt, rt) linearly.
  output2GateModule->Forward(*prevOutput, 
      output2GateModule->OutputParameter());
      

  // Merge the outputs(zt and rt).
  output = (input2GateModule->OutputParameter().submat(0, 0, 2 * outSize - 1, batchSize - 1) +
      output2GateModule->OutputParameter());

  // Pass the first outSize through inputGate(it).
  inputGateModule->Forward(output.submat(
      0, 0, 1 * outSize - 1, batchSize - 1), 
      inputGateModule->OutputParameter());

  // Pass the second through forgetGate.
  forgetGateModule->Forward(output.submat(
      1 * outSize, 0, 2 * outSize - 1, batchSize - 1),
      forgetGateModule->OutputParameter());

  InputType modInput = (forgetGateModule->OutputParameter() % *prevOutput);

  // Pass that through the outputHidden2GateModule.
  outputHidden2GateModule->Forward(modInput,
      outputHidden2GateModule->OutputParameter());

  // Merge for ot.
  OutputType outputH = input2GateModule->OutputParameter().submat(
      2 * outSize, 0, 3 * outSize - 1, batchSize - 1) +
      outputHidden2GateModule->OutputParameter();

  // Pass it through hiddenGate.
  hiddenStateModule->Forward(outputH,
      hiddenStateModule->OutputParameter());

  // Update the output (nextOutput): cmul1 + cmul2
  // Where cmul1 is input gate * prevOutput and
  // cmul2 is (1 - input gate) * hidden gate.
  output = (inputGateModule->OutputParameter()
      % (*prevOutput - hiddenStateModule->OutputParameter())) +
      hiddenStateModule->OutputParameter();

  forwardStep++;
  if (forwardStep == rho)
  {
    forwardStep = 0;
    if (!deterministic)
    {
      outParameter.emplace_back(allZeros.memptr(),
          allZeros.n_rows, allZeros.n_cols, false, true);
      prevOutput = --outParameter.end();
    }
    else
    {
      *prevOutput = OutputType(allZeros.memptr(),
          allZeros.n_rows, allZeros.n_cols, false, true);
    }
  }
  else if (!deterministic)
  {
    outParameter.push_back(output);
    prevOutput = --outParameter.end();
  }
  else
  {
    if (forwardStep == 1)
    {
      outParameter.clear();
      outParameter.push_back(output);

      prevOutput = outParameter.begin();
    }
    else
    {
      *prevOutput = output;
    }
  }
}

template<typename InputType, typename OutputType>
void GRUType<InputType, OutputType>::Backward(
    const InputType& input,
    const InputType& gy,
    OutputType& g)
{
  if (input.n_cols != batchSize)
  {
    batchSize = input.n_cols;
    prevError.resize(3 * outSize, batchSize);
    allZeros.zeros(outSize, batchSize);
    // Batch size better not change during an iteration...
    if (outParameter.size() > 1)
    {
      Log::Fatal << "GRU::Forward(): batch size cannot change during a "
          << "forward pass!" << std::endl;
    }

    outParameter.clear();
    outParameter.emplace_back(allZeros.memptr(),
        allZeros.n_rows, allZeros.n_cols, false, true);

    prevOutput = outParameter.begin();
    backIterator = outParameter.end();
    gradIterator = outParameter.end();
  }

  InputType gyLocal;
  if ((outParameter.size() - backwardStep  - 1) % rho != 0 && backwardStep != 0)
  {
    gyLocal = gy + output2GateModule->Delta();
  }
  else
  {
    gyLocal = InputType(((InputType&) gy).memptr(), gy.n_rows,
        gy.n_cols, false, false);
  }

  if (backIterator == outParameter.end())
  {
    backIterator = --(--outParameter.end());
  }

  // Delta zt.
  InputType dZt = gyLocal % (*backIterator -
      hiddenStateModule->OutputParameter());

  // Delta ot.
  InputType dOt = gyLocal % (arma::ones<InputType>(outSize, batchSize) -
      inputGateModule->OutputParameter());

  // Delta of input gate.
  inputGateModule->Backward(
      inputGateModule->OutputParameter(), dZt,
      inputGateModule->Delta());

  // Delta of hidden gate.
  hiddenStateModule->Backward(
      hiddenStateModule->OutputParameter(), dOt,
      hiddenStateModule->Delta());
      

  // Delta of outputHidden2GateModule.
  outputHidden2GateModule->Backward(
      outputHidden2GateModule->OutputParameter(),
      hiddenStateModule->Delta(),
      outputHidden2GateModule->Delta());

  // Delta rt.
  InputType dRt = outputHidden2GateModule->Delta() % *backIterator;

  // Delta of forget gate.
  forgetGateModule->Backward(
      forgetGateModule->OutputParameter(), dRt,
      forgetGateModule->Delta());

  // Put delta zt.
  prevError.submat(0, 0, 1 * outSize - 1, batchSize - 1) = 
      inputGateModule->Delta();

  // Put delta rt.
  prevError.submat(1 * outSize, 0, 2 * outSize - 1, batchSize - 1) =
      forgetGateModule->Delta();

  // Put delta ot.
  prevError.submat(2 * outSize, 0, 3 * outSize - 1, batchSize - 1) =
      hiddenStateModule->Delta();

  // Get delta ht - 1 for input gate and forget gate.
  InputType prevErrorSubview = prevError.submat(0, 0, 2 * outSize - 1,
      batchSize - 1);
  output2GateModule->Backward(
      input2GateModule->OutputParameter(),
      prevErrorSubview,
      output2GateModule->Delta());

  // Add delta ht - 1 from hidden state.
  output2GateModule->Delta() += outputHidden2GateModule->Delta() %
      forgetGateModule->OutputParameter();

  // Add delta ht - 1 from ht.
  output2GateModule->Delta() += gyLocal % 
      inputGateModule->OutputParameter();

  // Get delta input.
  input2GateModule->Backward(input2GateModule->OutputParameter(),
      prevError, input2GateModule->Delta());

  backwardStep++;
  backIterator--;

  g = input2GateModule->Delta();
}

template<typename InputType, typename OutputType>
void GRUType<InputType, OutputType>::Gradient(
    const InputType& input,
    const InputType& /* error */,
    OutputType& /* gradient */)
{
  if (input.n_cols != batchSize)
  {
    batchSize = input.n_cols;
    prevError.resize(3 * outSize, batchSize);
    allZeros.zeros(outSize, batchSize);
    // Batch size better not change during an iteration...
    if (outParameter.size() > 1)
    {
      Log::Fatal << "GRU::Forward(): batch size cannot change during a "
          << "forward pass!" << std::endl;
    }

    outParameter.clear();
    outParameter.emplace_back(allZeros.memptr(),
        allZeros.n_rows, allZeros.n_cols, false, true);

    prevOutput = outParameter.begin();
    backIterator = outParameter.end();
    gradIterator = outParameter.end();
  }

  if (gradIterator == outParameter.end())
  {
    gradIterator = --(--outParameter.end());
  }

  input2GateModule->Gradient(input, prevError, input2GateModule->Gradient());

  output2GateModule->Gradient(*gradIterator,
      prevError.submat(0, 0, 2 * outSize - 1, batchSize - 1),
      output2GateModule->Gradient());

  outputHidden2GateModule->Gradient(*gradIterator %
      forgetGateModule->OutputParameter(),
      prevError.submat(2 * outSize, 0, 3 * outSize - 1, batchSize - 1),
      outputHidden2GateModule->Gradient());

  gradIterator--;
}

template<typename InputType, typename OutputType>
void GRUType<InputType, OutputType>::ResetCell(const size_t /* size */)
{
  outParameter.clear();
  outParameter.emplace_back(allZeros.memptr(),
    allZeros.n_rows, allZeros.n_cols, false, true);

  prevOutput = outParameter.begin();
  backIterator = outParameter.end();
  gradIterator = outParameter.end();

  forwardStep = 0;
  backwardStep = 0;
}

template<typename InputType, typename OutputType>
template<typename Archive>
void GRUType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  // // If necessary, clean memory from the old model.
  // if (cereal::is_loading<Archive>())
  // {
  //   boost::apply_visitor(deleteVisitor, input2GateModule);
  //   boost::apply_visitor(deleteVisitor, output2GateModule);
  //   boost::apply_visitor(deleteVisitor, outputHidden2GateModule);
  //   boost::apply_visitor(deleteVisitor, inputGateModule);
  //   boost::apply_visitor(deleteVisitor, forgetGateModule);
  //   boost::apply_visitor(deleteVisitor, hiddenStateModule);
  // }

  // ar(CEREAL_NVP(inSize));
  // ar(CEREAL_NVP(outSize));
  // ar(CEREAL_NVP(rho));

  // ar(CEREAL_VARIANT_POINTER(input2GateModule));
  // ar(CEREAL_VARIANT_POINTER(output2GateModule));
  // ar(CEREAL_VARIANT_POINTER(outputHidden2GateModule));
  // ar(CEREAL_VARIANT_POINTER(inputGateModule));
  // ar(CEREAL_VARIANT_POINTER(forgetGateModule));
  // ar(CEREAL_VARIANT_POINTER(hiddenStateModule));
}

} // namespace ann
} // namespace mlpack

#endif
