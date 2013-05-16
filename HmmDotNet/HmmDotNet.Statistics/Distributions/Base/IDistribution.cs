using System;

namespace HmmDotNet.Statistics.Distributions
{
    public interface IDistribution : ICloneable , IEquatable<IDistribution>
    {
        /// <summary>
        ///     Relevant for descrite distibutions http:\\ynet.com
        /// </summary>
        /// <para>    
        ///   References:
        ///   <list type="bullet">
        ///     <item><description><a href="http://en.wikipedia.org/wiki/Probability_mass_function">Probability mass function</a></description></item>
        ///   </list></para>
        /// </remarks>
        /// <param name="x">Parameters needed to calcualte probability mass fanction</param>
        /// <returns></returns>
        double ProbabilityMassFunction(params double[] x);
        /// <summary>
        ///     Relevant for continuos distributions
        /// </summary>
        /// <para>    
        ///   References:
        ///   <list type="bullet">
        ///     <item><description><a href="http://en.wikipedia.org/wiki/Probability_density_function">Probability density function</a></description></item>
        ///   </list></para>
        /// </remarks>
        /// <param name="x">Parameters needed to calcualte probability density fanction</param>
        /// <returns></returns>
        double ProbabilityDensityFunction(params double[] x);
        /// <summary>
        ///     Evaluates distribution params according to the given observation list
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        IDistribution Evaluate(double[] observations);
        /// <summary>
        ///     Evaluates distribution params according to the given observation list
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="likelihood"></param>
        /// <returns></returns>
        IDistribution Evaluate(double[][] observations, out double likelihood);
        /// <summary>
        ///     Evaluates distribution params according to the given observation list, 
        ///     given set of weights for the multivariate distribution.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="weights"></param>
        /// <returns></returns>
        IDistribution Evaluate(double[] observations, double[] weights);
        /// <summary>
        ///     Evaluates distribution params according to the given observation list, 
        ///     given set of weights for the multivariate distribution.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="weights"></param>
        /// <param name="likelihood"></param>
        /// <returns></returns>
        IDistribution Evaluate(double[][] observations, double[] weights, out double likelihood);
    }
}
