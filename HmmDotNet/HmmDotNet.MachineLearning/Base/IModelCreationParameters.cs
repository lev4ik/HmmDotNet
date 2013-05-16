using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Base
{
    public interface IModelCreationParameters<TDistribution> where TDistribution : IDistribution
    {
        int? NumberOfStates { get; set; }
        int? Delta { get; set; }
        int? NumberOfComponents { get; set; }
        double[] Pi { get; set; }
        double[][] TransitionProbabilityMatrix { get; set; }
        TDistribution[] Emissions { get; set; }
    }
}