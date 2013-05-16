namespace HmmDotNet.MachineLearning.Algorithms
{
    public interface IVariableEstimatorCalculatorFactory
    {
        /// <summary>
        ///     Based upon T and K returns Variable Estimator Calculator
        /// </summary>
        /// <typeparam name="T">Distribution function type</typeparam>
        /// <typeparam name="K">Observation type</typeparam>
        /// <returns></returns>
        IVariablesEstimatorCalculator<T, K> GetCalculartor<T, K>();
    }
}
