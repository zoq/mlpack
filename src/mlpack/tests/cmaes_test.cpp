/**
 * @file cmaes_test.cpp
 * @author Marcus Edel
 *
 * Test the CMA-ES method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/cmaes/cmaes.hpp>
#include <mlpack/methods/reinforcement_learning/environment/cart_pole.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::rl;
using namespace mlpack::cmaes;

BOOST_AUTO_TEST_SUITE(CMAESTest);


class CartPoleANN
{
 public:
  CartPoleANN()
  {
    /*
     * Construct a feed forward network with 4 input nodes:
     * (position, velocity, angle, angular velocity),
     * 10 hidden nodes and 2 output nodes (forward, backward). The network
     * structure looks like:
     *
     *  Input         Hidden        Output
     *  Layer         Layer         Layer
     * +-----+       +-----+       +-----+
     * |     |       |     |       |     |
     * |     +------>|     +------>|     |
     * |     |     +>|     |     +>|     |
     * +-----+     | +--+--+     | +-----+
     *             |             |
     *  Bias       |  Bias       |
     *  Layer      |  Layer      |
     * +-----+     | +-----+     |
     * |     |     | |     |     |
     * |     +-----+ |     +-----+
     * |     |       |     |
     * +-----+       +-----+
     */
    model.Add<Linear<> >(4, 10);
    model.Add<SigmoidLayer<> >();
    model.Add<Linear<> >(10, 2);
    model.Add<LogSoftMax<> >();
  }

  //! Return the number of separable functions (the number of predictor points).
  size_t NumFunctions() const { return 1; }

  //! Evaluate a function.
  double Evaluate(const arma::mat& /* parameters */, const size_t /* i */)
  {
    // MountainCar defines "solving" as getting average reward of 195.0 over 100
    // consecutive trials. To keep it simple we just run 2 trails.
    double reward = 0;
    for (size_t trial = 0; trial < 2; ++trial)
    {
      // Create a new instance of the mountain car environment.
      CartPole env;
      state = env.InitialSample();

      while(!env.IsTerminal(state))
      {
        arma::mat response;
        model.Predict(state.Data(), response);
        const size_t actionIdx = arma::as_scalar(arma::find(
            arma::max(response.col(0)) == response.col(0), 1));

        // Get the reward using the current action and state.
        reward += env.Sample(state, CartPole::Action(actionIdx), nextState);
        state = nextState;
      }
    }

    reward /= 2;

    return reward;
  }

  //! Return the initial point for the optimization.
  const arma::mat& Parameters() const { return model.Parameters(); }
  //! Modify the initial point for the optimization.
  arma::mat& Parameters() { return model.Parameters(); }

 private:
  //! Matrix of (trained) parameters.
  arma::mat parameter;

  //! Locally-stored state.
  CartPole::State state;

  //! Locally-stored state.
  CartPole::State nextState;

  //! Locally-stored network.
  FFN<NegativeLogLikelihood<> > model;
};


BOOST_AUTO_TEST_CASE(SimpleCMAESTest)
{
  CartPoleANN function;
  CMAES<CartPoleANN> opt(function);
  opt.Optimize(function.Parameters());
}

BOOST_AUTO_TEST_SUITE_END();
