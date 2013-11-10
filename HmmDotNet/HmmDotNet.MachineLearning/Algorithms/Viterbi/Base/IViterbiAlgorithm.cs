using System.Collections.Generic;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public interface IViterbiAlgorithm
    {
        IList<IState> Run<TDistribution>(ISequenceData sequenceData, IHiddenMarkovModel<TDistribution> model) where TDistribution : IDistribution;
    }
}
