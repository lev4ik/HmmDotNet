using System;
using HmmDotNet.MachineLearning.Algorithms.Search.Base;

namespace HmmDotNet.MachineLearning.Algorithms.Search.Genetic
{
    public class Mutator : IMutatable
    {
        public Mutator(decimal mutationProbability)
        {
            MutationProbability = mutationProbability;
        }

        public decimal MutationProbability { get; private set; }

        public IChromosome<T> RunMutation<T>(IChromosome<T> x, decimal probability)
        {
            var representation = (IGene<T>[])x.Representation.Clone();
            if (MutationProbability >= probability)
            {
                var rd = new Random();
                var index = rd.Next(0, x.Representation.Length);
                representation[index].Mutate(probability);
            }
            return new Chromosome<T>(representation);
        }
    }
}
