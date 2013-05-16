namespace HmmDotNet.MachineLearning.Base
{
    public interface IPredictionResult
    {
        /// <summary>
        ///     Predicted values
        /// </summary>
        double[][] Predicted { get; set; }
    }
}
