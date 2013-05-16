using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Base
{
    public class ModelCreationParameters<TDistribution> : IModelCreationParameters<TDistribution> where TDistribution : IDistribution
    {
        public int? NumberOfStates { get; set; }
        public int? Delta { get; set; }
        public int? NumberOfComponents { get; set; }
        public double[] Pi { get; set; }
        public double[][] TransitionProbabilityMatrix { get; set; }
        public TDistribution[] Emissions { get; set; }
    }
}
