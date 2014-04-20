using System;
using System.Collections.Generic;

namespace HmmDotNet.MachineLearning.Algorithms.Search.Base
{
    public interface ISelector
    {
        Tuple<IChromosome<T>, IChromosome<T>> Selection<T>(IList<IChromosome<T>> population);
    }
}
