using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.GeneralPredictors.Base;

namespace HmmDotNet.MachineLearning.GeneralPredictors
{
    public class EvaluationRequest : IEvaluationRequest
    {
        public ErrorEstimatorType EstimatorType { get; set; }
        public IPredictionResult PredictionToEvaluate { get; set; }
        public IPredictionRequest PredictionParameters { get; set; }
    }
}
