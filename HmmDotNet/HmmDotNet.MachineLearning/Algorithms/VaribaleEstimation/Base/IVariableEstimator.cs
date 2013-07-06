namespace HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.Base
{
    public interface IVariableEstimator<out T, in P> where P : IEstimationParameters
    {
        T Estimate(P parameters);
    }
}
