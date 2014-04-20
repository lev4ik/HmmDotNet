using System;

namespace HmmDotNet.MachineLearning.Algorithms.Search.Base
{
    public interface IEvaluator
    {
        decimal Evaluate<T>(IChromosome<T> c);
    }
}
