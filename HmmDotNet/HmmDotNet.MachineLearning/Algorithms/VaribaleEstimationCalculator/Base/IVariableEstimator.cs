namespace HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator.Base
{
    public interface IVariableEstimator<out T>
    {
        T Estimate(bool normalized);
    }
}
