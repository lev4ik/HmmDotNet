namespace HmmDotNet.MachineLearning.GeneralPredictors
{
    public interface IPredictionErrorEstimator<T>
    {
        T[] CumulativeForecastError();
        T[] MeanError();
        T[] MeanSquaredError();
        T[] RootMeanSquaredError();
        T[] MeanAbsoluteDeviation();
        T[] MeanAbsolutePercentError();
        T ReturnOnInvestment();
    }
}
