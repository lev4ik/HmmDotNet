using System;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Base
{
    /// <summary>
    ///     Hidden Markov Model
    /// </summary>
    public interface IHiddenMarkovModelState<TDistribution> : ICloneable, IEquatable<IHiddenMarkovModelState<TDistribution>> where TDistribution : IDistribution
    {
        /// <summary>
        ///     Model Type
        /// </summary>
        ModelType Type { get; }
        /// <summary>
        ///     Initial state distribution
        /// </summary>
        double[] Pi { get; }
        /// <summary>
        ///     Transition probability matrix
        /// </summary>
        double[][] TransitionProbabilityMatrix { get; }
        /// <summary>
        ///     Emmission state function. Number of symbols per state will be derived from probability
        ///     distribution function
        /// </summary>
        TDistribution[] Emission { get; }
        /// <summary>
        ///     Number of components on Mixture distribution
        /// </summary>
        int C { get; }
        /// <summary>
        ///     Number of states
        /// </summary>
        int N { get; }
        /// <summary>
        ///     Number of observation symbols
        /// </summary>
        int M { get; }
        /// <summary>
        ///     Likelihood of getting this HMM state
        /// </summary>
        double Likelihood { get; set; }
        /// <summary>
        ///     Indicates that all calculations on the model will be performed with Log
        /// </summary>
        bool Normalized { get; set; }
    }
}
