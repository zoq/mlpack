/**
 * @file methods/ann/layer/recurrent_attention.hpp
 * @author Marcus Edel
 *
 * Definition of the RecurrentAttention class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RECURRENT_ATTENTION_HPP
#define MLPACK_METHODS_ANN_LAYER_RECURRENT_ATTENTION_HPP

#include <mlpack/prereqs.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

#include "layer_types.hpp"
#include "add_merge.hpp"
#include "sequential.hpp"
#include "layer.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class implements the Recurrent Model for Visual Attention, using a
 * variety of possible layer implementations.
 *
 * For more information, see the following paper.
 *
 * @code
 * @article{MnihHGK14,
 *   title   = {Recurrent Models of Visual Attention},
 *   author  = {Volodymyr Mnih, Nicolas Heess, Alex Graves, Koray Kavukcuoglu},
 *   journal = {CoRR},
 *   volume  = {abs/1406.6247},
 *   year    = {2014},
 *   url     = {https://arxiv.org/abs/1406.6247}
 * }
 * @endcode
 *
 * @tparam  InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class RecurrentAttentionType: public Layer<InputType, OutputType>
{
 public:
  /**
   * Default constructor: this will not give a usable RecurrentAttention object,
   * so be sure to set all the parameters before use.
   */
  RecurrentAttentionType();

  /**
   * Create the RecurrentAttention object using the specified modules.
   *
   * @param outSize The module output size.
   * @param rnn The recurrent neural network module.
   * @param action The action module.
   * @param rho Maximum number of steps to backpropagate through time (BPTT).
   */
  template<typename RNNModuleType, typename ActionModuleType>
  RecurrentAttentionType(const size_t outSize,
                     const RNNModuleType& rnn,
                     const ActionModuleType& action,
                     const size_t rho);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */

  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */

  void Backward(const InputType& /* input */,
                const OutputType& gy,
                OutputType& g);

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param * (input) The input parameter used for calculating the gradient.
   * @param * (error) The calculated error.
   * @param * (gradient) The calculated gradient.
   */

  void Gradient(const InputType& /* input */,
                const OutputType& /* error */,
                OutputType& /* gradient */);

  //! Clone the RecurrentAttentionType object. This handles polymorphism
  //  correctly.
  RecurrentAttentionType* Clone() const
      { return new RecurrentAttentionType(*this); }

  //! Get the model modules.
  std::vector<Layer<InputType, OutputType> *>& Model() { return network; }

    //! The value of the deterministic parameter.
  bool Deterministic() const { return deterministic; }
  //! Modify the value of the deterministic parameter.
  bool& Deterministic() { return deterministic; }

  //! Get the parameters.
  OutputType const& Parameters() const { return parameters; }
  //! Modify the parameters.
  OutputType& Parameters() { return parameters; }

  //! Get the output parameter.
  OutputType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputType& Delta() { return delta; }

  //! Get the gradient.
  OutputType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputType& Gradient() { return gradient; }

  //! Get the module output size.
  size_t OutSize() const { return outSize; }

  //! Get the number of steps to backpropagate through time.
  size_t const& Rho() const { return rho; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Calculate the gradient of the attention module.
  void IntermediateGradient()
  {
    intermediateGradient.zeros();

    // Gradient of the action module.
    if (backwardStep == (rho - 1))
    {
          actionModule->Gradient(initialInput, actionError)
    }
    else
    {
          actionModule->Gradient(actionModule->outputParameter(),actionError);
    }

    // Gradient of the recurrent module.
    rnnModule->Gradient(rnnModule->outputParameter(), recurrentError);

    attentionGradient += intermediateGradient;
  }

  //! Locally-stored module output size.
  size_t outSize;

  //! Locally-stored start module.
  Layer<InputType, OutputType> rnnModule;

  //! Locally-stored input module.
  Layer<InputType, OutputType> actionModule;

  //! Number of steps to backpropagate through time (BPTT).
  size_t rho;

  //! Locally-stored number of forward steps.
  size_t forwardStep;

  //! Locally-stored number of backward steps.
  size_t backwardStep;

  //! If true dropout and scaling is disabled, see notes above.
  bool deterministic;

  //! Locally-stored weight object.
  OutputType parameters;

  //! Locally-stored model modules.
  std::vector<Layer<InputType, OutputType>*> network;

  //! Locally-stored feedback output parameters.
  std::vector<OutputType> feedbackOutputParameter;

  //! List of all module parameters for the backward pass (BBTT).
  std::vector<OutputType> moduleOutputParameter;

  //! Locally-stored delta object.
  OutputType delta;

  //! Locally-stored gradient object.
  OutputType gradient;

  //! Locally-stored output parameter object.
  OutputType outputParameter;

  //! Locally-stored recurrent error parameter.
  OutputType recurrentError;

  //! Locally-stored action error parameter.
  OutputType actionError;

  //! Locally-stored action delta.
  OutputType actionDelta;

  //! Locally-stored recurrent delta.
  OutputType rnnDelta;

  //! Locally-stored initial action input.
  InputType initialInput;

  //! Locally-stored attention gradient.
  OutputType attentionGradient;

  //! Locally-stored intermediate gradient for the attention module.
  OutputType intermediateGradient;
}; // class RecurrentAttention

} // namespace ann
} // namespace mlpack

typedef RecurrentAttentionType<arma::mat, arma::mat> RecurrentAttention;

// Include implementation.
#include "recurrent_attention_impl.hpp"

#endif
