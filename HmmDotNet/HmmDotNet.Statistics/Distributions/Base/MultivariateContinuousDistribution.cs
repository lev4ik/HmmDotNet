using HmmDotNet.Mathematic.Extentions;

namespace HmmDotNet.Statistics.Distributions
{
    /// <summary>
    ///     Continuous Multivariate distribution is defined by it's mean vector Mean[] and covariance matrix
    ///     Covariance[,] that hold covariance for each Xi and Xj in, random variables in this multivariate distribution.
    /// <para>    
    ///   References:
    ///   <list type="bullet">
    ///     <item><description><a href="http://en.wikipedia.org/wiki/Multivariate_random_variable">Multivariate random variable</a></description></item>
    ///   </list></para>
    /// </remarks>
    /// </summary>
    public abstract class MultivariateContinuousDistribution : IMultivariateDistribution
    {
        #region Members

        private int _dimension;

        #endregion Members

        #region IMultivariateDistribution Members
        /// <summary>
        ///     Vector of variances for each random variables in th multivariate distribution
        /// </summary>
        public abstract double[] Variance { get; }
        /// <summary>
        ///     Mean vector of each random variable in the multivariate distribution
        /// </summary>
        public abstract double[] Mean { get; }
        /// <summary>
        ///     Covariance matrix for each couple of random variables X and Y in the multivariate distribution
        /// </summary>
        public abstract double[,] Covariance { get; }
        /// <summary>
        ///     Number random variables in the multivariate distribution
        /// </summary>
        public int Dimension 
        { 
            get 
            {
                return _dimension;
            } 
        }

        #endregion

        #region IDistribution Members

        public double ProbabilityMassFunction(params double[] x)
        {
            throw new System.NotImplementedException();
        }

        public abstract double ProbabilityDensityFunction(params double[] x);

        public abstract IDistribution Evaluate(double[] observations);

        public abstract IDistribution Evaluate(double[] observations, double[] weights);

        public abstract IDistribution Evaluate(double[][] observations, out double likelihood);

        public abstract IDistribution Evaluate(double[][] observations, double[] weights, out double likelihood);

        #endregion

        #region ICloneable Members

        public abstract object Clone();

        #endregion

        #region Constructors

        protected MultivariateContinuousDistribution(int dimension)
        {
            _dimension = dimension;
        }

        #endregion Constructors

        public bool Equals(IDistribution other)
        {
            var d = other as MultivariateDiscreteDistribution;
            if (d == null)
            {
                //throw new ApplicationException("Incompatable types, other is not UnivariateDiscreteDistribution");
                return false;
            }

            return VectorExtentions.EqualsTo(Variance, d.Variance) &&
                   VectorExtentions.EqualsTo(Mean, d.Mean) &&
                   Covariance.EqualsTo(d.Covariance) &&
                   Dimension == d.Dimension;
        }
    }
}
