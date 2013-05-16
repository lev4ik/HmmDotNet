namespace HmmDotNet.MachineLearning.Base
{
    public class PredictionResultEvaluation : IEvaluationResult
    {
       public double[] CumulativeForecastError { get; set; }
       public double[] MeanError { get; set; }
       public double[] MeanSquaredError { get; set; }
       public double[] RootMeanSquaredError { get; set; }
       public double[] MeanAbsoluteDeviation { get; set; }
       public double[] MeanAbsolutePercentError { get; set; }
       public double ReturnOnInvestment { get; set; }
    }
}