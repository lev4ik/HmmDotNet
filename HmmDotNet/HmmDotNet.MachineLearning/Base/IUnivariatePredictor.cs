namespace HmmDotNet.MachineLearning.Base
{
    public interface IUnivariatePredictor<T> where T : struct
    {
        /// <summary>
        ///     Predicts single value based on observation vector
        /// </summary>
        /// <param name="observations">Vector</param>
        /// <returns></returns>
        PredictionResult Predict(T[] observations);
        /// <summary>
        ///     Predicts single value based on observation vector
        /// </summary>
        /// <param name="observations">Vector</param>
        /// <param name="weights">Vector</param>
        /// <returns>Prediction value</returns>
        PredictionResult Predict(T[] observations, T[] weights);
    }
}
