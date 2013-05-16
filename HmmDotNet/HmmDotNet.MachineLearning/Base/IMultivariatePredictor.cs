namespace HmmDotNet.MachineLearning.Base
{
    public interface IMultivariatePredictor<T> where T : struct
    {
        /// <summary>
        ///     Predicts vector value based on observation matrix
        /// </summary>
        /// <param name="observations">Matrix</param>
        /// <param name="weights">Vector</param>
        /// <returns>Prediction vector</returns>
        PredictionResult Predict(T[][] observations, T[] weights);
        /// <summary>
        ///     Predicts vector value based on observation matrix
        /// </summary>
        /// <param name="observations">Matrix</param>
        /// <param name="weights">Vector</param>
        /// <param name="numberOfDays">Number of prediction to make</param>
        /// <returns></returns>
        PredictionResult Predict(T[][] observations, T[] weights, int numberOfDays);
        /// <summary>
        ///     Predicts vector value based on training and test sequence
        /// </summary>
        /// <param name="training"></param>
        /// <param name="test"></param>
        /// <param name="weights"></param>
        /// <param name="numberOfDays"></param>
        /// <returns></returns>
        PredictionResult Predict(T[][] training, T[][] test,T[] weights, int numberOfDays);
        /// <summary>
        ///     Prediction results evaluation with MAPE function
        /// </summary>
        /// <param name="predicted">Set of predicted values</param>
        /// <param name="observed">Matrix of values that was actualy observed</param>
        /// <returns></returns>
        double[] EvaluatePrediction(PredictionResult predicted, T[][] observed);
    }
}
