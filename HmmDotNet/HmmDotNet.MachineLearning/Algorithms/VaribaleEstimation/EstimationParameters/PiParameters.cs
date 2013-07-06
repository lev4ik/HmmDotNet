using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.Base;

namespace HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters
{
    public class PiParameters : IEstimationParameters
    {
        public bool Normalized { get; set; }

        public double[][] Gamma { get; set; }

        public int N { get; set; }
    }
}
