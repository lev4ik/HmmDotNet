using HmmDotNet.MachineLearning.GeneralPredictors.Base;

namespace HmmDotNet.MachineLearning.Base
{
    public interface IEvaluationRequest
    {
        /// <summary>
        ///     Type of actual error estimator
        /// </summary>
        ErrorEstimatorType EstimatorType { get; set; }
        /// <summary>
        ///     Predcited values
        /// </summary>
        IPredictionResult PredictionToEvaluate { get; set; }
        /// <summary>
        ///     Parameters that was used to predict the values
        /// </summary>
        IPredictionRequest PredictionParameters { get; set; }
    }
}
