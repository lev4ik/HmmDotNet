using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator.Base;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator.EstimationParameters
{
    public class AlphaBetaTransitionProbabiltyMatrixParameters<TDistribution> : IEstimationParameters where TDistribution : IDistribution
    {
        public bool Normalized { get; set; }

        public double[] Weights { get; set; }

        public double[][] Observations { get; set; }

        public double[][] Alpha { get; set; }

        public double[][] Beta { get; set; }

        public IHiddenMarkovModel<TDistribution> Model { get; set; } 
    }
}
