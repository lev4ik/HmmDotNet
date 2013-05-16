using System.Collections.Generic;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public interface IParameterEstimations<TDistribution> where TDistribution : IDistribution
    {
        int L { get; }

        double[][] Alpha { get; }

        double[][] Beta { get; }

        IHiddenMarkovModelState<TDistribution> Model { get; }       

        double[][] Coefficients { get; }

        IList<IObservation> Observation { get; }
    }
}
