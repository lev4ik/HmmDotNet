namespace HmmDotNet.MachineLearning
{
    public interface IPredictor<T>
    {
        /// <summary>
        ///     Predicts single value based on observation vector
        /// </summary>
        /// <param name="observations">Vector</param>
        /// <param name="weights">Vector</param>
        /// <returns>Prediction value</returns>
        T Predict(T[] observations, T[] weights);
        /// <summary>
        ///     Predicts vector value based on observation matrix
        /// </summary>
        /// <param name="observations">Matrix</param>
        /// <param name="weights">Vector</param>
        /// <returns>Prediction vector</returns>
        T[] Predict(T[][] observations, T[] weights);
    }
}
