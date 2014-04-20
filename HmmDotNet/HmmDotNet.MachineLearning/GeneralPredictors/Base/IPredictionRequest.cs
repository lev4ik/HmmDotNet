using System.Collections.Generic;

namespace HmmDotNet.MachineLearning.Base
{
    public interface IPredictionRequest
    {
        /// <summary>
        ///     Number of days to predict
        /// </summary>
        int NumberOfDays { get; set; }
        /// <summary>
        ///     Prediction tolerance
        /// </summary>
        double Tolerance { get; set; }
        /// <summary>
        ///     Training set for the predictor
        /// </summary>
        double[][] TrainingSet { get; set; }
        /// <summary>
        ///     Test set for the predictor
        /// </summary>
        double[][] TestSet { get; set; }

        int NumberOfTrainingIterations { get; set; }

        double TrainingLikelihoodTolerance { get; set; }
        /// <summary>
        ///     Parameters specific for current algorithm instance
        /// </summary>
        IDictionary<string, string> AlgorithmSpecificParameters { get; set; }
        /// <summary>
        ///     Validates that AlgorithmSpecificParameters was filled correctly
        /// </summary>
        /// <returns></returns>
        bool ValidateAlgorithmSpecificParameters();
    }
}
