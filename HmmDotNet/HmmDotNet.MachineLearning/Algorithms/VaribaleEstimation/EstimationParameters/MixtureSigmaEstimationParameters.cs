using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters
{
    public class MixtureSigmaEstimationParameters<TDistribution> : MixtureCoefficientEstimationParameters<TDistribution>
                                                                   where TDistribution : IDistribution
    {
        public double[,][] Mu { get; set; }
    }
}
