/**
 * @file trish_test.cpp
 * @author Marcus Edel
 *
 * Test file for the TRish optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/trish/trish.hpp>

#include <mlpack/core/optimizers/problems/sphere_function.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace arma;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

using namespace mlpack::distribution;
using namespace mlpack::regression;

using namespace mlpack;

BOOST_AUTO_TEST_SUITE(TRishTest);

/**
 * Run TRish on logistic regression and make sure the results are acceptable.
 */
BOOST_AUTO_TEST_CASE(TRishLogisticRegressionTest)
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(3, 1000);
  arma::Row<size_t> responses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Shuffle the dataset.
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0,
      data.n_cols - 1, data.n_cols));
  arma::mat shuffledData(3, 1000);
  arma::Row<size_t> shuffledResponses(1000);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    shuffledData.col(i) = data.col(indices[i]);
    shuffledResponses[i] = responses[indices[i]];
  }

  // Create a test set.
  arma::mat testData(3, 1000);
  arma::Row<size_t> testResponses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    testData.col(i) = g1.Random();
    testResponses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    testData.col(i) = g2.Random();
    testResponses[i] = 1;
  }

  LogisticRegressionFunction<> lrFunction(shuffledData,
      shuffledResponses, 0.5);


  // The value G gauges the magnitude of the stochastic gradient estimates,
  // which depends on problem scaling. Compute G during an initial SG phase
  // before starting TRish.
  arma::mat gradient;
  arma::mat parameters = arma::rowvec(shuffledData.n_rows + 1, arma::fill::zeros);
  lrFunction.Gradient(parameters, 0, gradient, 1);
  const double G = arma::norm(gradient);

  TRish optimizer(0.1, 1, 4 / G, 1 / (2 * G), 500000, 1e-9, true);
  LogisticRegression<> lr(shuffledData, shuffledResponses, optimizer, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  BOOST_REQUIRE_CLOSE(acc, 100.0, 0.3); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  BOOST_REQUIRE_CLOSE(testAcc, 100.0, 0.6); // 0.6% error tolerance.
}

/**
 * Test the TRish optimizer on the Sphere function.
 */
BOOST_AUTO_TEST_CASE(EVESphereFunctionTest)
{
  SphereFunction f(2);
  arma::mat coordinates = f.GetInitialPoint();

  // The value G gauges the magnitude of the stochastic gradient estimates,
  // which depends on problem scaling. Compute G during an initial SG phase
  // before starting TRish.
  arma::mat gradient;
  f.Gradient(coordinates, 0, gradient, 2);
  const double G = arma::norm(gradient);

  TRish optimizer(0.1, 1, 4 / G, 1 / (2 * G), 500000, 1e-9, true);
  optimizer.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(coordinates[0], 0.1);
  BOOST_REQUIRE_SMALL(coordinates[1], 0.1);
}

BOOST_AUTO_TEST_SUITE_END();
