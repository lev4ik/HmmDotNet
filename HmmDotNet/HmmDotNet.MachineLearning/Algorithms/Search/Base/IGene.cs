using System;

namespace HmmDotNet.MachineLearning.Algorithms.Search.Base
{
    public interface IGene<T> : ICloneable
    {
        T[] Representation { get; }

        void Mutate(decimal probability);
    }
}
