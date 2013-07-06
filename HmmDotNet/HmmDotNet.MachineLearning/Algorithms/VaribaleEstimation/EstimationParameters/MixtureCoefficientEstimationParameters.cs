using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters
{
    public class MixtureCoefficientEstimationParameters<TDistribution> : MixtureAdvancedEstimationParameters<TDistribution>
                                                                         where TDistribution : IDistribution
    {
        public double[][,] GammaComponents { get; set; }

        public double[][] Gamma { get; set; }
    }
}
