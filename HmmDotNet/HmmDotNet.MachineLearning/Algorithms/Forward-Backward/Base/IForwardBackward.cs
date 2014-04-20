using System.Collections.Generic;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public interface IForwardBackward
    {
        bool Normalized { get; set; }
        double[][] Alpha { get; }
        double[][] Beta { get; }
        double RunForward<TEmmisionType>(IList<IObservation> observations, IHiddenMarkovModel<TEmmisionType> model) where TEmmisionType : IDistribution;
        double RunBackward<TEmmisionType>(IList<IObservation> observations, IHiddenMarkovModel<TEmmisionType> model) where TEmmisionType : IDistribution;
    }
}