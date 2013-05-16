using System;
using HmmDotNet.MachineLearning.GeneralPredictors.Base;

namespace HmmDotNet.MachineLearning.GeneralPredictors
{
    public static class PredictionErrorEstimatorFactory
    {
        public static IPredictionErrorEstimator<double> GetErrorEstimator(ErrorEstimatorType estimatorType, double[][] actual, double[][] predicted)
        {
            switch (estimatorType)
            {
                case ErrorEstimatorType.Trend:
                    return new TrendPredictionErrorEstimator(actual, predicted);
                case ErrorEstimatorType.Value:
                    return new ValuePredictionErrorEstimator(actual, predicted);
            }
            throw new ApplicationException("Not implemented type");
        }
    }
}
