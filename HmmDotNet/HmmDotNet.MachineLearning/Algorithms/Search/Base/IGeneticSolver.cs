using System;
using System.Collections.Generic;

namespace HmmDotNet.MachineLearning.Algorithms.Search.Base
{
    public interface IGeneticSolver
    {
        /// <summary>
        /// 1.  Calculate the fitness f(x) of each chromosome x in the population.
        /// 2.  Repeat the following steps until n offspring have been created:
        ///     (a) Select a pair of parent chromosomes from the current population, the
        ///         probability of selection increasing as a function of fitness.
        ///     (b) Crossover with probability pc (the crossover probability), pair by
        ///         taking part of the chromosome from one parent and the other part from
        ///         the other parent. This forms a single offspring.
        ///     (c) Mutate the resulting offspring at each locus with probability pm (the
        ///         mutation probability) and place the resulting chromosome in the new
        ///         population. Mutation typically replaces the current value of a locus
        ///         (e.g., 0) with another value (e.g., 1).
        /// 3. Replace the current population with the new population.    
        /// 4. Return to step 1.
        /// </summary>
        IList<IChromosome<T>> Solve<T>(IList<IChromosome<T>> population);
        /// <summary>
        ///     Checks if the algorithm has convirged
        /// </summary>
        /// <returns></returns>
        bool Convirged();
    }
}
