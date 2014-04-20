using System;

namespace HmmDotNet.MachineLearning.Algorithms.Search.Base
{
    public interface IChromosome<T> : ICloneable
    {
        IGene<T>[] Representation  { get; }
        decimal FintnessValue { get; set; }
    }
}
