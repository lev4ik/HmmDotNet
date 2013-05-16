namespace HmmDotNet.Statistics.Distributions
{
    public interface IMultivariateDistribution : IDistribution
    {
        double[] Variance { get; }
        double[] Mean { get; }
        double[,] Covariance { get; }
        int Dimension { get; }
    }
}
