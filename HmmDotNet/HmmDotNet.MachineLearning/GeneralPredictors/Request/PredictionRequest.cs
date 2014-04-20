using System.Collections.Generic;
using HmmDotNet.MachineLearning.Base;

namespace HmmDotNet.MachineLearning.GeneralPredictors
{
    public class PredictionRequest : IPredictionRequest
    {
        public int NumberOfDays { get; set; }
        public double Tolerance { get; set; }
        public double[][] TrainingSet { get; set; }
        public double[][] TestSet { get; set; }
        public int NumberOfTrainingIterations { get; set; }
        public double TrainingLikelihoodTolerance { get; set; }
        public IDictionary<string, string> AlgorithmSpecificParameters { get; set; }

        public bool ValidateAlgorithmSpecificParameters()
        {
            return true;
        }
    }
}
