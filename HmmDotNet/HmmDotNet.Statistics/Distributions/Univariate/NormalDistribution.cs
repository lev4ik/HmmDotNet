using System;

namespace HmmDotNet.Statistics.Distributions.Univariate
{
    public class NormalDistribution : UnivariateContinuousDistribution
    {
        #region Private Members & Constants

        private static readonly double SqrtPi = Math.Sqrt(2 * Math.PI);

        private double _mean;
        private double _variance;

        #endregion Private Members

        #region Properties

        public override double Variance
        {
            get { return _variance; }
        }

        public override double Mean
        {
            get { return _mean; }
        }

        #endregion Properties

        #region Constructors

        /// <summary>
        ///     Create Standart Gaussian Distribution with mean zero and variance one 
        /// </summary>
        public NormalDistribution() : 
            this(0.0d, 1.0d)
        {
            
        }

        public NormalDistribution(double mean) : 
            this(mean, 1.0)
        {
            
        }

        /// <summary>
        ///     Base constructor that will initialize Normal Distribution with given mean and variance
        /// </summary>
        /// <param name="mean">Mean value of the distribution</param>
        /// <param name="variance">Variance of the distribution</param>
        public NormalDistribution(double mean, double variance)
        {
            _mean = mean;
            _variance = variance;
        }

        #endregion Constructors

        #region Methods

        /// <summary>
        ///     ZScore of given observation x
        /// </summary>
        /// <param name="x">Observation Value</param>
        /// <returns></returns>
        public double ZScore(double x)
        {
            return (x - _mean)/_variance;
        }

        /// <summary>
        ///     Computes Probability Density Function (PDF) of x[0] observation with mean and variance calculated above
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public override double ProbabilityDensityFunction(params double[] x)
        {
            var exponentPower = -Math.Pow(ZScore(x[0]), 2) / (2 * _variance);
            var p = (1d / (SqrtPi * _variance)) * Math.Exp(exponentPower);
            if (p > 1 || p < 0)
            {
                throw new ApplicationException(string.Format("Probability function value {0} must be in range [0..1]", p));
            }
            return p;
        }

        /// <summary>
        ///     We will calculate maximum likelihood of observing the observations sequence.
        ///     In this calculation we will calculate the maximum likely mean and variance for the evaluated distribution. This is true because when 
        ///     calculating maximum likelihood we can see that the maximized value of the mean and variance is the mean and variance calculated from the observation sequence.
        /// </summary>
        /// <param name="observations">Observation vector</param>
        /// <returns></returns>
        public override IDistribution Evaluate(double[] observations)
        {
            var mean = Utils.Mean(observations);
            var variance = Utils.Variance(observations, mean);
            return new NormalDistribution(mean, variance);
        }

        public override IDistribution Evaluate(double[] observations, double[] weights)
        {
            var mean = Utils.Mean(observations, weights);
            var variance = Utils.Variance(observations, weights, mean);
            return new NormalDistribution(mean, variance);
        }

        public override IDistribution Evaluate(double[][] observations, out double likelihood)
        {
            throw new NotImplementedException("Not multivariate distribution");
        }

        public override IDistribution Evaluate(double[][] observations, double[] weights, out double likelihood)
        {
            throw new NotImplementedException("Not multivariate distribution");                    
        }

        public override object Clone()
        {
            return new NormalDistribution(_mean, _variance);
        }

        #endregion Methods
    }
}
