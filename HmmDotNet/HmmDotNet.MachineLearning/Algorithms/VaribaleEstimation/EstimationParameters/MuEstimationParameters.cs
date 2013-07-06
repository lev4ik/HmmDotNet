using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters
{
    public class MuEstimationParameters<TDistribution> : BasicEstimationParameters<TDistribution> 
                                                              where TDistribution : IDistribution
    {
        public double[][] Gamma { get; set; }
    }
}
