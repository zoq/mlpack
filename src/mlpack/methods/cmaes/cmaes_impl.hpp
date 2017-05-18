/**
 * @file cmaes_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Covariance Matrix Adaptation Evolution Strategy.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_CMAES_CMAES_IMPL_HPP
#define MLPACK_METHODS_CMAES_CMAES_IMPL_HPP

// In case it hasn't been included yet.
#include "cmaes.hpp"

namespace mlpack {
namespace cmaes {

template<typename DecomposableFunctionType>
CMAES<DecomposableFunctionType>::CMAES(DecomposableFunctionType& function) :
    function(function)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename DecomposableFunctionType>
double CMAES<DecomposableFunctionType>::Optimize(arma::mat& iterate)
{
  // while(!evo.testForTermination() && !task.Success())
  // {
  //   // Generate lambda new search points, sample population
  //   pop = evo.samplePopulation();

  //   // evaluate the new search points using fitness function from above
  //   for (int i = 0; i < evo.sampleSize(); ++i)
  // {
  //   //wights and flush
  //    neuralNet.Flush();

  //    setWeights(linkGenes, pop[i], (int) evo.dimension());

  //    arFunvals[i] = task.EvalFitness(neuralNet);

  //   }
  //   // update the search distribution used for sampleDistribution()
  //   evo.updateDistribution(arFunvals);
  // }



  // I basically removed all CMA-ES methods and just show the basic idea of the
  // evaluation part as shown above.

  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // To keep track of where we are and how things are going.
  size_t currentFunction = 0;
  double overallObjective = 0;

  // Calculate the first objective function.
  for (size_t i = 0; i < numFunctions; ++i)
    overallObjective += function.Evaluate(iterate, i);

  std::cout << "First overall objective: " << overallObjective << std::endl;

  const size_t maxIterations = 4;
  for (size_t i = 1; i != maxIterations; ++i, ++currentFunction)
  {
    // Is this iteration the start of a sequence?
    if ((currentFunction % numFunctions) == 0)
    {
      currentFunction = 0;
    }

    // Generate lambda new search points, sample population
    // pop = samplePopulation();

    // Evaluate the new search points using fitness function from above.
    const size_t sampleSize = 2;
    for (size_t sample = 0; sample < sampleSize; ++sample)
    {
      // setWeights(linkGenes, pop[i], (int) evo.dimension());
      // Update function/network parameters/weights.
      // The part where you used setWeights(...), since iterate is a matrix,
      // we basically do the same, in this case I use random weights, but you
      // can iterate through the matrix with a for loop to set the weights
      // according to the CMA-ES parameter.
      iterate.randn();

      // arFunvals[i] = task.EvalFitness(neuralNet);
      // Evaluates the given function with the given parameters/iterate.
      const double fitness = function.Evaluate(iterate, 0);
      std::cout << "Fitness: " << fitness << std::endl;
    }
    // Update the search distribution used for sampleDistribution().
    // updateDistribution(arFunvals);
  }

  // Calculate final objective.
  overallObjective = 0;
  for (size_t i = 0; i < numFunctions; ++i)
    overallObjective += function.Evaluate(iterate, i);
  return overallObjective;
}

} // namespace cmaes
} // namespace mlpack

#endif
