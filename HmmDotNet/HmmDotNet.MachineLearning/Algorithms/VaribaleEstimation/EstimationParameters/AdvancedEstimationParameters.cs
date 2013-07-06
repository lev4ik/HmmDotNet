using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters
{
    public class AdvancedEstimationParameters<TDistribution> : BasicEstimationParameters<TDistribution> where TDistribution : IDistribution
    {
        public double[][] Alpha { get; set; }

        public double[][] Beta { get; set; }
    }
}
