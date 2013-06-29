using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator.Base;

namespace HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator.EstimationParameters
{
    public class PiParameters : IEstimationParameters
    {
        public bool Normalized { get; set; }

        public double[][] Gamma { get; set; }

        public int N { get; set; }
    }
}
