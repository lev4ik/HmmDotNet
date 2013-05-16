using System.Diagnostics;
using HmmDotNet.Mathematic;
using System;
using HmmDotNet.Mathematic.Distance;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Algorithms;

namespace HmmDotNet.Statistics.Distributions.Multivariate
{
    /// <summary>
    ///     Multivariate Gaussian distribution
    /// </summary>
    [Serializable]
    public class NormalDistribution : MultivariateContinuousDistribution
    {
        #region Private Members

        private const double MINIMUM_PROBABILITY = 10e-100;
        private readonly double SqrtPi = Math.Sqrt(2 * Math.PI);
        private double[] _mean;
        private double[,] _covariance;
        private double[] _variance;

        private Matrix _matrix;

        #endregion Private Members

        #region Properties

        public override double[] Variance
        {
            get { return _variance; }
        }

        public override double[] Mean
        {
            get { return _mean; }
        }

        public override double[,] Covariance
        {
            get { return _covariance; }
        }

        #endregion Properties

        #region Constructors

        public NormalDistribution(int dimension) : base(dimension)
        {
            
        }

        public NormalDistribution(double[] mean, double[,] covariance) : this(mean.Length)
        {
            _matrix = new Matrix(covariance);
            _mean = mean;
            _covariance = covariance;
            _variance = _matrix.Diagonal(); 
        }

        #endregion Constructors

        #region Methods

        /// <summary>
        ///     1. Calculate Mahalanobis Distance for exponent factor
        ///     2. Check if the matrix is semetric
        ///     3. Check if the matrix is positive and semidifine
        ///     4. Calculate Inverse matrix
        ///     5. Check if matrix is Positive definite
        /// </summary>
        /// <param name="x">Observation vector</param>
        /// <returns>Probability</returns>
        public override double ProbabilityDensityFunction(params double[] x)
        {
            var k = x.Length;
            var sigmaInverse = _matrix.Inverse();
            var distance = Mahalanobis.Calculate(x, Mean, sigmaInverse);
            var constant = (1 / Math.Pow(SqrtPi, k)) * (1 / Math.Sqrt(_matrix.Determinant));

            var p = constant * Math.Exp(-0.5 * distance);
            // This means that the probability is to low (underflow)
            if (p.EqualsToZero() || double.IsNaN(p))
            {
                p = MINIMUM_PROBABILITY;
            }
//            if (p > 1 || p < 0)
//            {
//                Debug.Write(Environment.NewLine + "Covariance matrix " + Environment.NewLine);
//                Debug.Write(_matrix.ToString());
//                Debug.Write(Environment.NewLine + "Mean Vector " + Environment.NewLine);
//                Debug.Write((new Vector(Mean)).ToString());
//                Debug.Write(Environment.NewLine + "Observation Vector " + Environment.NewLine);
//                Debug.Write((new Vector(x)).ToString());
//                Debug.Write(Environment.NewLine);
//                //throw new ApplicationException(string.Format("Probability function value {0} must be in range [0..1]", p));
//            }
            return p > 1 ? 1 : p;
        }

        /// <summary>
        ///     Fits Distribution parameters to given observation vector
        /// </summary>
        /// <param name="observations">Vector</param>
        /// <param name="likelihood"></param>
        /// <returns>IDistribution</returns>
        public override IDistribution Evaluate(double[][] observations, out double likelihood)
        {
            var rows = observations.Length;
            var weights = new double[rows];
            for (var i = 0; i < observations.Length; i++)
            {
                weights[i] = 1d / rows;
            }

            return Evaluate(observations, weights, out likelihood);
        }

        /// <summary>
        ///     Fits Distribution parameters to given observation vector
        /// </summary>
        /// <param name="observations">Vector</param>
        /// <param name="weights">Vector</param>
        /// <param name="likelihood">Output value</param>
        /// <returns>IDistribution</returns>
        public override IDistribution Evaluate(double[][] observations, double[] weights, out double likelihood)
        {
            var mean = Utils.Mean(observations, weights);
            var covariance = Utils.Covariance(observations, mean, weights);
            var result = new NormalDistribution(mean, covariance);
            likelihood = LogLikelihood.Calculate(observations, result.Covariance, result.Mean);
            return result;
        }

        public override IDistribution Evaluate(double[] observations)
        {
            throw new NotImplementedException("Not univariate distribution");
        }

        public override IDistribution Evaluate(double[] observations, double[] weights)
        {
            throw new NotImplementedException("Not univariate distribution");
        }

        /// <summary>
        ///     Clone current distribution
        /// </summary>
        /// <returns>Normal Distribution</returns>
        public override object Clone()
        {
            return new NormalDistribution(_mean, _covariance);
        }

        #endregion Methods
    }
}
