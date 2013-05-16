using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.MachineLearning.Base
{
    public interface IGaussianMixtureModelState
    {
        /// <summary>
        ///     Gets Model's underlying mixture
        /// </summary>
        Mixture<IMultivariateDistribution> Mixture { get; }
    }
}
