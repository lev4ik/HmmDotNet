using System;
using HmmDotNet.MachineLearning.Algorithms.Search.Base;

namespace HmmDotNet.MachineLearning.Algorithms.Search.Genetic
{
    public class Crossover : ICrossover
    {
        public Crossover(decimal crossoverProbability)
        {
            CrossoverProbability = crossoverProbability;
        }

        public decimal CrossoverProbability { get; private set; }

        public IChromosome<T> RunCrossover<T>(IChromosome<T> x, IChromosome<T> y, decimal probability)
        {
            var representation = (IGene<T>[])x.Representation.Clone();
            if (CrossoverProbability >= probability)
            {
                var rd = new Random();
                var index = rd.Next(0, x.Representation.Length - 1);
                var crossovered = new IGene<T>[x.Representation.Length];
                Array.Copy(representation, crossovered, index + 1);
                Array.Copy(y.Representation, index + 1, crossovered, index + 1, x.Representation.Length - (index + 1));

                return new Chromosome<T>(crossovered);
            }
            return new Chromosome<T>(representation);
        }
    }
}
