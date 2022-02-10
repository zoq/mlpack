/**
 * @file methods/ann/layer/lstm.hpp
 * @author Marcus Edel
 *
 * Definition of the LSTM class, which implements a LSTM network layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LSTM_HPP
#define MLPACK_METHODS_ANN_LAYER_LSTM_HPP

#include <mlpack/prereqs.hpp>
#include <limits>

#include "layer.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the LSTM module class.
 * The implementation corresponds to the following algorithm:
 *
 * @f{eqnarray}{
 * i &=& sigmoid(W \cdot x + W \cdot h + W \cdot c + b) \\
 * f &=& sigmoid(W  \cdot x + W \cdot h + W \cdot c + b) \\
 * z &=& tanh(W \cdot x + W \cdot h + b) \\
 * c &=& f \odot c + i \odot z \\
 * o &=& sigmoid(W \cdot x + W \cdot h + W \cdot c + b) \\
 * h &=& o \odot tanh(c)
 * @f}
 *
 * Note that if an LSTM layer is desired as the first layer of a neural network,
 * an IdentityLayer should be added to the network as the first layer, and then
 * the LSTM layer should be added.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Graves2013,
 *   author  = {Alex Graves and Abdel{-}rahman Mohamed and Geoffrey E. Hinton},
 *   title   = {Speech Recognition with Deep Recurrent Neural Networks},
 *   journal = CoRR},
 *   year    = {2013},
 *   url     = {http://arxiv.org/abs/1303.5778},
 * }
 * @endcode
 *
 * \see FastLSTM for a faster LSTM version which combines the calculation of the
 * input, forget, output gates and hidden state in a single step.
 *
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
<<<<<<< Updated upstream
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class LSTM : public Layer<InputType, OutputType>
=======
template<
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class LSTMType : public Layer<InputType, OutputType>
>>>>>>> Stashed changes
{
 public:
  //! Create the LSTM object.
  LSTMType();

  /**
   * Create the LSTM layer object using the specified parameters.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   * @param rho Maximum number of steps to backpropagate through time (BPTT).
   */
  LSTMType(const size_t inSize,
           const size_t outSize,
           const size_t rho = std::numeric_limits<size_t>::max());

  //! Clone the LSTMType object. This handles polymorphism correctly.
  LSTMType* Clone() const { return new LSTMType(*this); }

  /**
   * Reset the layer parameter. The method is called to
   * assign the allocated memory to the internal learnable parameters.
   */
  void SetWeights(typename OutputType::elem_type* weightsPtr);

  /**
   * Ordinary feed-forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed-forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   * @param cellState Cell state of the LSTM.
   * @param useCellState Use the cellState passed in the LSTM cell.
   */
  void Forward(const InputType& input,
               OutputType& output,
               OutputType& cellState,
               bool useCellState = false);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& input,
                const OutputType& gy,
                OutputType& g);
<<<<<<< Updated upstream

  /*
   * Reset the layer parameter.
   */
  void Reset();
=======
>>>>>>> Stashed changes

  /*
   * Resets the cell to accept a new input. This breaks the BPTT chain starts a
   * new one.
   *
   * @param size The current maximum number of steps through time.
   */
  void ResetCell(const size_t size);

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const InputType& input,
                const OutputType& error,
                OutputType& gradient);

  //! Get the maximum number of steps to backpropagate through time (BPTT).
  size_t Rho() const { return rho; }
  //! Modify the maximum number of steps to backpropagate through time (BPTT).
  size_t& Rho() { return rho; }

<<<<<<< Updated upstream
  //! Get the parameters.
  OutputType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputType& Parameters() { return weights; }

  //! Get the output parameter.
  OutputType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputType& Delta() { return delta; }

  //! Get the gradient.
  OutputType const& Gradient() const { return grad; }
  //! Modify the gradient.
  OutputType& Gradient() { return grad; }

  //! Get the number of input units.
  size_t InSize() const { return inSize; }

  //! Get the number of output units.
  size_t OutSize() const { return outSize; }
=======
    //! Get the parameters.
  const OutputType& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputType& Parameters() { return weights; }

  //! Get the weight of the layer.
  OutputType const& Weight() const { return weight; }
  //! Modify the weight of the layer.
  OutputType& Weight() { return weight; }
>>>>>>> Stashed changes

  const size_t WeightSize() const
  {
<<<<<<< Updated upstream
    // TODO ...
=======
    return (4 * outSize * inSize + 7 * outSize + 4 * outSize * outSize);
  }

  void ComputeOutputDimensions()
  {
    inSize = std::accumulate(this->inputDimensions.begin(),
        this->inputDimensions.end(), 0);
    this->outputDimensions = std::vector<size_t>(this->inputDimensions.size(),
        1);

    // The Linear layer flattens its input.
    this->outputDimensions[0] = outSize;
>>>>>>> Stashed changes
  }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Number of steps to backpropagate through time (BPTT).
  size_t rho;

  //! Locally-stored number of forward steps.
  size_t forwardStep;

  //! Locally-stored number of backward steps.
  size_t backwardStep;

  //! Locally-stored number of gradient steps.
  size_t gradientStep;

  //! Locally-stored weight object.
  OutputType weights;

  //! Locally-stored previous output.
  OutputType prevOutput;

  //! Locally-stored batch size.
  size_t batchSize;

  //! Current batch step, alias for batchSize - 1.
  size_t batchStep;

  //! Current gradient step to keep track of the backpropagate through time
  //! step.
  size_t gradientStepIdx;

  //! Locally-stored cell activation error.
  OutputType cellActivationError;
<<<<<<< Updated upstream
=======

  //! Locally-stored delta object.
  OutputType delta;

  //! Locally-stored gradient object.
  OutputType grad;

  //! Locally-stored output parameter object.
  OutputType outputParameter;
>>>>>>> Stashed changes

  //! Weights between the output and input gate.
  OutputType output2GateInputWeight;

  //! Weights between the input and gate.
  OutputType input2GateInputWeight;

  //! Bias between the input and input gate.
  OutputType input2GateInputBias;

  //! Weights between the cell and input gate.
  OutputType cell2GateInputWeight;

  //! Weights between the output and forget gate.
  OutputType output2GateForgetWeight;

  //! Weights between the input and gate.
  OutputType input2GateForgetWeight;

  //! Bias between the input and gate.
  OutputType input2GateForgetBias;

  //! Bias between the input and gate.
  OutputType cell2GateForgetWeight;

  //! Weights between the output and gate.
  OutputType output2GateOutputWeight;

  //! Weights between the input and gate.
  OutputType input2GateOutputWeight;

  //! Bias between the input and gate.
  OutputType input2GateOutputBias;

  //! Weights between cell and output gate.
  OutputType cell2GateOutputWeight;

  //! Locally-stored input gate parameter.
  OutputType inputGate;

  //! Locally-stored forget gate parameter.
  OutputType forgetGate;

  //! Locally-stored hidden layer parameter.
  OutputType hiddenLayer;

  //! Locally-stored output gate parameter.
  OutputType outputGate;

  //! Locally-stored input gate activation.
  OutputType inputGateActivation;

  //! Locally-stored forget gate activation.
  OutputType forgetGateActivation;

  //! Locally-stored output gate activation.
  OutputType outputGateActivation;

  //! Locally-stored hidden layer activation.
  OutputType hiddenLayerActivation;

  //! Locally-stored input to hidden weight.
  OutputType input2HiddenWeight;

  //! Locally-stored input to hidden bias.
  OutputType input2HiddenBias;

  //! Locally-stored output to hidden weight.
  OutputType output2HiddenWeight;

  //! Locally-stored cell parameter.
  OutputType cell;

  //! Locally-stored cell activation error.
  OutputType cellActivation;

  //! Locally-stored forget gate error.
  OutputType forgetGateError;

  //! Locally-stored output gate error.
  OutputType outputGateError;

  //! Locally-stored previous error.
  OutputType prevError;

  //! Locally-stored output parameters.
  OutputType outParameter;

  //! Locally-stored input cell error parameter.
  OutputType inputCellError;

  //! Locally-stored input gate error.
  OutputType inputGateError;

  //! Locally-stored hidden layer error.
  OutputType hiddenError;

  //! Locally-stored current rho size.
  size_t rhoSize;

  //! Current backpropagate through time steps.
  size_t bpttSteps;
}; // class LSTM

// Convenience typedefs.

// Standard LSTM layer.
typedef LSTMType<arma::mat, arma::mat, NoRegularizer> LSTM;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "lstm_impl.hpp"

#endif
