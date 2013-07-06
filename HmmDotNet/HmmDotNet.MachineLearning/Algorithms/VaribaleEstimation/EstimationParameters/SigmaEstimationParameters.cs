using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters
{
    public class SigmaEstimationParameters<TDistribution, TMean> : MuEstimationParameters<TDistribution>
                                                                   where TDistribution : IDistribution
    {
        public SigmaEstimationParameters()
        {
            
        }

        public SigmaEstimationParameters(MuEstimationParameters<TDistribution> @params)
        {
            Gamma = @params.Gamma;
            Model = @params.Model;
            Normalized = @params.Normalized;
            Observations = @params.Observations;
        }

        public TMean Mean { get; set; }
    }
}
