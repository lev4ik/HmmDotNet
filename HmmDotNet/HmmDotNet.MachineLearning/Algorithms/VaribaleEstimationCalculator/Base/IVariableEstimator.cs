namespace HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator.Base
{
    public interface IVariableEstimator<out T, in P> where P : IEstimationParameters
    {
        T Estimate(P parameters);
    }
}
