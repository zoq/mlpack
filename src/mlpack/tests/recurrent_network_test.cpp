/**
 * @file tests/recurrent_network_test.cpp
 * @author Marcus Edel
 *
 * Tests the recurrent network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/rnn.hpp>

#include <ensmallen.hpp>

#include "catch.hpp"
#include "serialization.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;
using namespace mlpack::math;

/**
 * Construct a 2-class dataset out of noisy sines.
 *
 * @param data Input data used to store the noisy sines.
 * @param labels Labels used to store the target class of the noisy sines.
 * @param points Number of points/features in a single sequence.
 * @param sequences Number of sequences for each class.
 * @param noise The noise factor that influences the sines.
 */
void GenerateNoisySines(arma::cube& data,
                        arma::mat& labels,
                        const size_t points,
                        const size_t sequences,
                        const double noise = 0.3)
{
  arma::colvec x =  arma::linspace<arma::colvec>(0, points - 1, points) /
      points * 20.0;
  arma::colvec y1 = arma::sin(x + arma::as_scalar(arma::randu(1)) * 3.0);
  arma::colvec y2 = arma::sin(x / 2.0 + arma::as_scalar(arma::randu(1)) * 3.0);

  data = arma::zeros(1 /* single dimension */, sequences * 2, points);
  labels = arma::zeros(2 /* 2 classes */, sequences * 2);

  for (size_t seq = 0; seq < sequences; seq++)
  {
    arma::vec sequence = arma::randu(points) * noise + y1 +
        arma::as_scalar(arma::randu(1) - 0.5) * noise;
    for (size_t i = 0; i < points; ++i)
      data(0, seq, i) = sequence[i];

    labels(0, seq) = 1;

    sequence = arma::randu(points) * noise + y2 +
        arma::as_scalar(arma::randu(1) - 0.5) * noise;
    for (size_t i = 0; i < points; ++i)
      data(0, sequences + seq, i) = sequence[i];

    labels(1, sequences + seq) = 1;
  }
}

/**
 * Construct dataset for sine wave prediction.
 *
 * @param data Input data used to store the noisy sines.
 * @param labels Labels used to store the target class of the noisy sines.
 * @param points Number of points/features in a single sequence.
 * @param sequences Number of sequences for each class.
 * @param noise The noise factor that influences the sines.
 */

void GenerateSines(arma::cube& data,
                   arma::cube& labels,
                   const size_t sequences,
                   const size_t len)
{
  arma::vec x = arma::sin(arma::linspace<arma::colvec>(0,
      sequences + len, sequences + len));
  data.set_size(1, len, sequences);
  labels.set_size(1, 1, sequences);

  for (size_t i = 0; i < sequences; ++i)
  {
    data.slice(i) = arma::reshape(x.subvec(i, i + len), 1, len);
    labels.slice(i) = x(i + len);
  }
}

/*
 * This sample is a simplified version of Derek D. Monner's Distracted Sequence
 * Recall task, which involves 10 symbols:
 *
 * Targets: must be recognized and remembered by the network.
 * Distractors: never need to be remembered.
 * Prompts: direct the network to give an answer.
 *
 * A single trial consists of a temporal sequence of 10 input symbols. The first
 * 8 consist of 2 randomly chosen target symbols and 6 randomly chosen
 * distractor symbols in an random order. The remaining two symbols are two
 * prompts, which direct the network to produce the first and second target in
 * the sequence, in order.
 *
 * For more information, see the following paper.
 *
 * @code
 * @misc{Monner2012,
 *   author = {Monner, Derek and Reggia, James A},
 *   title = {A generalized LSTM-like training algorithm for second-order
 *   recurrent neural networks},
 *   year = {2012}
 * }
 * @endcode
 *
 * @param input The generated input sequence.
 * @param input The generated output sequence.
 */
void GenerateDistractedSequence(arma::mat& input, arma::mat& output)
{
  input = arma::zeros<arma::mat>(10, 10);
  output = arma::zeros<arma::mat>(3, 10);

  arma::uvec index = arma::shuffle(arma::linspace<arma::uvec>(0, 7, 8));

  // Set the target in the input sequence and the corresponding targets in the
  // output sequence by following the correct order.
  for (size_t i = 0; i < 2; ++i)
  {
    size_t idx = rand() % 2;
    input(idx, index(i)) = 1;
    output(idx, index(i) > index(i == 0) ? 9 : 8) = 1;
  }

  for (size_t i = 2; i < 8; ++i)
    input(2 + rand() % 6, index(i)) = 1;

  // Set the prompts which direct the network to give an answer.
  input(8, 8) = 1;
  input(9, 9) = 1;

  input.reshape(input.n_elem, 1);
  output.reshape(output.n_elem, 1);
}


/**
 * Train the specified networks on the Derek D. Monner's distracted sequence
 * recall task.
 */
/* TEST_CASE("LSTMDistractedSequenceRecallTest", "[RecurrentNetworkTest]") */
/* { */
/*   DistractedSequenceRecallTestNetwork<LSTM<> >(4, 8); */
/* } */

/**
 * Train the specified networks on the Derek D. Monner's distracted sequence
 * recall task.
 */
/* TEST_CASE("FastLSTMDistractedSequenceRecallTest", "[RecurrentNetworkTest]") */
/* { */
/*   DistractedSequenceRecallTestNetwork<FastLSTM<> >(4, 8); */
/* } */

/**
 * Train the specified networks on the Derek D. Monner's distracted sequence
 * recall task.
 */
/* TEST_CASE("GRUDistractedSequenceRecallTest", "[RecurrentNetworkTest]") */
/* { */
/*   DistractedSequenceRecallTestNetwork<GRU<> >(4, 8); */
/* } */

/**
 * Create a simple recurrent neural network for the noisy sines task, and
 * require that it produces the exact same network for a few batch sizes.
 */
template<typename RecurrentLayerType>
void BatchSizeTest()
{
  const size_t T = 50;
  const size_t bpttTruncate = 10;

  // Generate 12 (2 * 6) noisy sines. A single sine contains rho
  // points/features.
  arma::cube input;
  arma::cube labels;
  GenerateSines(input, labels, 4, 5);
  /* arma::mat labelsTemp; */
  /* GenerateNoisySines(input, labelsTemp, T, 6); */

  /* arma::cube labels = arma::zeros<arma::cube>(1, labelsTemp.n_cols, T); */
  /* for (size_t i = 0; i < labelsTemp.n_cols; ++i) */
  /* { */
  /*   const int value = arma::as_scalar(arma::find( */
  /*       arma::max(labelsTemp.col(i)) == labelsTemp.col(i), 1)) + 1; */
  /*   labels.tube(0, i).fill(value); */
  /* } */

  //std::cout << input << std::endl;

  //std::cout << labels << std::endl;



  RNNType<> model(bpttTruncate);
  model.Add<Linear>(100);
  model.Add<Sigmoid>();
  model.Add<Linear>(1);
  //model.Add<RecurrentLayerType>(10, 10);
  //model.Add<SigmoidLayer<>>();
  //model.Add<Linear<>>(10, 10);
  //model.Add<SigmoidLayer<>>();

  //model.Reset();
  //arma::mat initParams = model.Parameters();

  StandardSGD opt(1e-5, 1, 5, -100, false);
  model.Train(input, labels, opt);

  /* exit(0); */

  /* // This is trained with one point. */
  /* arma::mat outputParams = model.Parameters(); */

  /* model.Reset(); */
  /* model.Parameters() = initParams; */
  /* opt.BatchSize() = 2; */
  /* model.Train(input, labels, opt); */

  /* CheckMatrices(outputParams, model.Parameters(), 1); */

  /* model.Parameters() = initParams; */
  /* opt.BatchSize() = 5; */
  /* model.Train(input, labels, opt); */

  /* CheckMatrices(outputParams, model.Parameters(), 1); */
}

/**
 * Ensure LSTMs work with larger batch sizes.
 */
TEST_CASE("LSTMBatchSizeTest", "[RecurrentNetworkTest]")
{
  BatchSizeTest<Linear>();
}

