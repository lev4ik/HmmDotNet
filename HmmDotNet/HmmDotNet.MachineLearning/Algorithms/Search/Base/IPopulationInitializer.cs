using System.Collections.Generic;

namespace HmmDotNet.MachineLearning.Algorithms.Search.Base
{
    public interface IPopulationInitializer
    {
        IList<IChromosome<T>> Initialize<T>(int populationSize, int chromosomeSize, decimal mutationProbability);
    }
}
