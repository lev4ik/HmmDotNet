using System.Collections.Generic;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator.Base;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator.EstimationParameters
{
    public class BasicEstimationParameters<TDistribution> : IEstimationParameters where TDistribution : IDistribution
    {
        public bool Normalized { get; set; }

        public IHiddenMarkovModel<TDistribution> Model { get; set; }

        public IList<IObservation> Observations { get; set; }
    }
}
