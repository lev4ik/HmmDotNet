using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    /// <summary>
    ///     BaumWelch algorithm definition
    /// </summary>
    public interface IBaumWelchAlgorithm<TDistribution> where TDistribution : IDistribution
    {
        bool Normalized { get; set; }
        /// <summary>
        ///     Run BaumWelch algorithm and returns calculated estimations for HMM parameters
        ///     Run until the algorithms convired, both HMM's are equesl or while there is a change in P(O|HMM1) > P(O|HMM)
        /// </summary>
        /// <returns></returns>
        IHiddenMarkovModelState<TDistribution> Run(int maxIterations, double likelihoodTolerance);       
    }
}