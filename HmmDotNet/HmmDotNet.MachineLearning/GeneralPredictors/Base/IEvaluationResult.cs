namespace HmmDotNet.MachineLearning.Base
{
    public interface IEvaluationResult
    {
        double[] CumulativeForecastError { get; set; }
        double[] MeanError { get; set; }
        double[] MeanSquaredError { get; set; }
        double[] RootMeanSquaredError { get; set; }
        double[] MeanAbsoluteDeviation { get; set; }
        double[] MeanAbsolutePercentError { get; set; }
        double ReturnOnInvestment { get; set; }
    }
}
