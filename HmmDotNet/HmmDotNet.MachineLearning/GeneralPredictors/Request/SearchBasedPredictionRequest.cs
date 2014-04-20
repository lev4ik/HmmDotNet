using System.Collections.Generic;
using HmmDotNet.MachineLearning.Base;

namespace HmmDotNet.MachineLearning.GeneralPredictors
{
    public class SearchBasedPredictionRequest : IPredictionRequest
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
            if (AlgorithmSpecificParameters == null)
            {
                return false;
            }

            if (!AlgorithmSpecificParameters.ContainsKey("NumberOfSamplePoints"))
            {
                return false;
            }

            if (!AlgorithmSpecificParameters.ContainsKey("NumberOfWinningPoints"))
            {
                return false;
            }

            if (!AlgorithmSpecificParameters.ContainsKey("NumberOfPredictionIterations"))
            {
                return false;
            }

            if (!AlgorithmSpecificParameters.ContainsKey("Epsilon"))
            {
                return false;
            }

            return true;
        }
    }
}
