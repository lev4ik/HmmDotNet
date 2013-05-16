using System;
using System.Diagnostics;
using HmmDotNet.Mathematic;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Algorithms;

namespace HmmDotNet.Statistics.Distributions.Multivariate
{
    /// <summary>
    ///     Mixture of Multivariate Continuous Distributions. Mixture distribution
    ///     consists of a finite number , m , distributions and a mixing distribution
    ///     that will select of of the components.
    /// </summary>
    /// <typeparam name="T">Multivariate Continuous Distributions</typeparam>
    [Serializable]
    public class Mixture<T> : MultivariateContinuousDistribution where T : IMultivariateDistribution
    {
        #region Private Methods

        private readonly double[] _coefficients;
        private T[] _components;

        private double[] _mean;
        private double[,] _covariance;
        private double[] _variance;
        private int _stepsTillConvirgence;

        #endregion Private Methods

        #region Properties

        public int StepsTillConvirgence
        {
            get
            {
                return _stepsTillConvirgence;
            }
        }

        public double[] Coefficients
        {
            get
            {
                return _coefficients;
            }
        }

        public T[] Components
        {
            get
            {
                return _components;
            }
        }

        public override double[] Variance
        {
            get
            {
                if (_variance == null)
                {
                    var m = new Matrix(Covariance);
                    _variance = m.Diagonal();
                }
                return _variance;
            }
        }

        public override double[] Mean
        {
            get
            {
                if (_mean == null)
                {
                    _mean = new double[Dimension];
                    for (var i = 0; i < Dimension; i++)
                    {
                        for (var j = 0; j < _components.Length; j++)
                        {
                            _mean[i] += _coefficients[j] * _components[j].Mean[i];
                        }
                    }
                }
                return _mean;
            }
        }

        /// <summary>
        ///     Calculation of the covariance matrix will be done based on
        ///     Var[X]=E[Var[X|Y]]+Var[E[X|Y]]
        /// </summary>
        public override double[,] Covariance
        {
            get
            {
                if (_covariance == null)
                {
                    _covariance = new double[Dimension, Dimension];
                    var evar = new double[Dimension, Dimension];
                    for (var i = 0; i < Dimension; i++)
                    {
                        for (var j = 0; j < Dimension; j++)
                        {
                            for (var k = 0; k < _components.Length; k++)
                            {
                                evar[i, j] = evar[i, j] + _components[k].Covariance[i, j];
                            }
                        }
                    }

                    var means = new double[_components.Length][];
                    for (int k = 0; k < _components.Length; k++)
                    {
                        means[k] = _components[k].Mean;
                    }
                    var vare = Utils.Covariance(means);

                    _covariance = MatrixExtentions.Add(evar, vare);
                }
                return _covariance;
            }
        }

        #endregion Properties

        #region Constructors

        public Mixture(int components, int dimentions) : base(dimentions)
        {
            _components = new T[components];
            _coefficients = new double[components];
            for (var i = 0; i < _coefficients.Length; i++)
            {
                _coefficients[i] = 1.0 / _coefficients.Length;
            }
        }
        /// <summary>
        ///     Base constructor that will distribute the coefficients uniformly between components
        /// </summary>
        /// <param name="components"></param>
        public Mixture(params T[] components) : base(components[0].Dimension)
        {
            _components = components;
            _coefficients = new double[components.Length];
            for (var i = 0; i < _coefficients.Length; i++)
            {
                _coefficients[i] = 1.0 / _coefficients.Length;
            }
        }

        public Mixture(double[] coefficients, params T[] components) : base(components[0].Dimension)
        {
            _components = components;
            _coefficients = coefficients;
        }

        #endregion Constructors

        #region Methods

        /// <summary>
        ///     Calculates PDF given observation x. Posterior probability p(i|x)=[p(i)*p(x|i)]/p(x).
        ///     This probability is also called responsability of component [i] to produce observation [x].
        /// </summary>
        /// <param name="i">Component number</param>
        /// <param name="x">Observation point</param>
        /// <returns></returns>
        public double ProbabilityDensityFunction(int i, params double[] x)
        {
            var p = (_coefficients[i] * _components[i].ProbabilityDensityFunction(x)) / ProbabilityDensityFunction(x);
            if (p > 1 || p < 0 )
            {
                Debug.WriteLine("Coefficient {0} , component probability {1} , model probability {2}", _coefficients[i] , _components[i].ProbabilityDensityFunction(x) , ProbabilityDensityFunction(x));
            }
            return p > 1 ? 1 : p;
        }

        /// <summary>
        ///     Calculates PDF given observation x.
        ///     PDF is calculating by summming over all components PDF's
        ///     at point x and normilized by it's coefficient.
        /// </summary>
        /// <param name="x">Observation point</param>
        /// <returns></returns>
        public override double ProbabilityDensityFunction(params double[] x)
        {
            var p = 0.0d;
            for (var i = 0; i < _components.Length; i++)
            {
                p += _coefficients[i] * _components[i].ProbabilityDensityFunction(x);
            }
//            if (p > 1 || p < 0)
//            {
//                Debug.WriteLine("Probability {0}", p);
//            }
            return p > 1 ? 1 : p;
        }

        #region Evaluate

        public override IDistribution Evaluate(double[] observations)
        {
            throw new NotImplementedException("Not univariate distribution");
        }

        public override IDistribution Evaluate(double[] observations, double[] weights)
        {
            throw new NotImplementedException("Not univariate distribution");
        }

        public override IDistribution Evaluate(double[][] observations, out double likelihood)
        {
            var weights = new double[observations.Length];
            for (var i = 0; i < observations.Length; i++)
            {
                weights[i] = 1.0d / observations.Length;
            }

            return Evaluate(observations, weights, 1e-3, out likelihood);
        }

        public override IDistribution Evaluate(double[][] observations, double[] weights, out double likelihood)
        {
            return Evaluate(observations, weights, 1e-3, out likelihood);
        }

        public IDistribution Evaluate(double[][] observations, double[] weights, double treshold, out double likelihood)
        {
            // Calculate Maximization of the Likelihood
            _stepsTillConvirgence = 0;
            var convirged = false;
            var pdf = (IMultivariateDistribution[])_components.Clone();
            var K = _coefficients.Length;
            var N = observations.Length;
            
            //// Initialize pi, covariance matrix and mean 
            var pi = (double[])_coefficients.Clone();
            weights = weights.Product(N);

            likelihood = LogLikelihood.Calculate(observations, _coefficients, GetCovariances(pdf), GetMeans(pdf));

            while (!convirged)
            {
                //// Expectation
                var gamma = CalculateGamma(observations, weights, pdf, pi);
                //// Maximization
                pi = CalculatePi(gamma);
                //// For each component train new distribution function
                for (var k = 0; k < K; k++)
                {
                    var mean = CalculateMean(gamma, observations, k);
                    var covariance = CalculateCovariance(gamma, observations, mean, k);
                    pdf[k] = new NormalDistribution(mean, covariance);
                }            

                //// Check treshold
                var newLikelihood = LogLikelihood.Calculate(observations, _coefficients, GetCovariances(pdf), GetMeans(pdf));
                if (double.IsNaN(newLikelihood) || double.IsInfinity(newLikelihood))
                {
                    throw new ApplicationException("EM algorithm does not convirged");
                }

                convirged = (newLikelihood - likelihood) <= treshold;
                likelihood = newLikelihood;
                _stepsTillConvirgence++;
            }

            return new Mixture<IMultivariateDistribution>(pi, pdf);
        }

        private static double[][,] GetCovariances(IMultivariateDistribution[] pdfs)
        {
            var covariances = new double[pdfs.Length][,];

            for (int i = 0; i < pdfs.Length; i++)
            {
                covariances[i] = pdfs[i].Covariance;
            }

            return covariances;
        }

        public static double[][] GetMeans(IMultivariateDistribution[] pdfs)
        {
            var means = new double[pdfs.Length][];

            for (int i = 0; i < pdfs.Length; i++)
            {
                means[i] = pdfs[i].Mean;
            }

            return means;            
        }

        private static double[,] CalculateCovariance(double[][] gamma, double[][] observations, double[] mean, int k)
        {
            var K = observations[0].Length;
            var N = observations.Length;

            var covariance = new double[K, K];
            var nK = gamma[k].Sum();
            for (var n = 0; n < N; n++)
            {
                var x = observations[n];

                var z = x.Substruct(mean[k]);
                var m = z.OuterProduct(z);
                m = m.Product(gamma[k][n]);
                covariance = covariance.Add(m);
            }
            return covariance.Product(1 / nK);
        }

        private static double[] CalculateMean(double[][] gamma, double[][] observations, int k)
        {
            var K = observations[0].Length;
            var N = observations.Length;
            var mean = new double[K];
            var nK = gamma[k].Sum();
            for (var n = 0; n < N; n++)
            {
                mean = mean.Add(observations[n].Product(gamma[k][n]));
            }
            
            return mean.Product(1 / nK);
        }

        private static double[] CalculatePi(double[][] gamma)
        {
            var K = gamma.GetLength(0);
            var N = gamma[0].GetLength(0);
            var pi = new double[K];
            for (var k = 0; k < K; k++)
            {
                pi[k] = gamma[k].Sum() / N;
            }

            return pi;
        }

        private static double[][] CalculateGamma(double[][] observations, double[] weights, IMultivariateDistribution[] pdf, double[] pi)
        {
            var K = pi.Length;
            var N = observations.Length;
            var gamma = new double[K][];
            for (var k = 0; k < K; k++)
            {
                gamma[k] = new double[N];
            }

            var nominator = new double[K];

            for (var n = 0; n < N; n++)
            {
                var denominator = 0.0d;
                var observation = observations[n];
                var w = weights[n];
                for (var k = 0; k < K; k++)
                {
                    denominator += nominator[k] = pi[k] * pdf[k].ProbabilityDensityFunction(observation) * w;
                }

                for (var k = 0; k < K; k++)
                {
                    gamma[k][n] = nominator[k] / denominator;
                }                
            }
            
            return gamma;
        }

        #endregion Evaluate

        public override object Clone()
        {
            return new Mixture<T>((double[])_coefficients.Clone(), (T[])_components.Clone());
        }

        #endregion Methods
    }
}
