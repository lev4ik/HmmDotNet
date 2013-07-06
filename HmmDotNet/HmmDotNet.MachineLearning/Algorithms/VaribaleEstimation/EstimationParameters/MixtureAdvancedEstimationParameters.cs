using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters
{
    public class MixtureAdvancedEstimationParameters<TDistribution> : AdvancedEstimationParameters<TDistribution>
                                                                      where TDistribution : IDistribution
    {
        public int L { get; set; }
    }
}
