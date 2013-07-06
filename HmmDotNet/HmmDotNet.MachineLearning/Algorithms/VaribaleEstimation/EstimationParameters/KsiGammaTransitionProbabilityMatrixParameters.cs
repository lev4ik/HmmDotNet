using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.Base;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters
{
    public class KsiGammaTransitionProbabilityMatrixParameters<TDistribution> : IEstimationParameters where TDistribution : IDistribution
    {
        public bool Normalized { get; set; }

        public double[][] Gamma { get; set; }

        public double[][,] Ksi { get; set; }

        public IHiddenMarkovModel<TDistribution> Model { get; set; }

        public int T { get; set; }
    }
}
