namespace HmmDotNet.Statistics.Distributions
{
    public interface IUnivariateDistribution : IDistribution
    {
        double Variance { get; }
        double Mean { get; }
    }
}
