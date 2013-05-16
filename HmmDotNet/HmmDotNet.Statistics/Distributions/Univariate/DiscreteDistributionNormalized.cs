using System;
using HmmDotNet.Mathematic.Extentions;

namespace HmmDotNet.Statistics.Distributions.Univariate
{
    public class DiscreteDistributionNormalized : UnivariateDiscreteDistribution
    {
        #region Members

        private double[] _symbols;

        private double[] _probabilities;

        private double _mean;

        private double _variance;

        #endregion Members

        public double[] Symbols
        {
            get { return _symbols; }
        }

        public double[] Probbilities
        {
            get { return _probabilities; }
        }

        public override double Variance
        {
            get
            {
                if(_mean == 0)
                {
                    _mean = 0d;
                    for (var i = 0; i < _symbols.Length; i++)
                    {
                        _mean += _symbols[i]*_probabilities[i];
                    }
                }
                _variance = 0;
                for (var i = 0; i < _symbols.Length; i++)
                {
                    _mean += Math.Pow(_symbols[i] - _mean, 2) * _probabilities[i];
                }
                return _variance;
            }
        }

        public override double Mean
        {
            get
            {
                _mean = 0d;
                for (var i = 0; i < _symbols.Length; i++)
                {
                    _mean += _symbols[i] * _probabilities[i];
                }
                return _mean;
            }
        }

        public DiscreteDistributionNormalized(params double[][] observations)
        {
            if (observations.Length < 2)
            {
                throw new ArgumentException("Supply two arrays");
            }
            _symbols = observations[0];
            _probabilities = observations[1];
        }

        public override double ProbabilityMassFunction(params double[] x)
        {
            var observationValue = x[0];
            for (var i = 0; i < _symbols.Length; i++)
            {                
                if (observationValue == _symbols[i])
                {
                    return _probabilities[i];
                }
            }
            return double.NaN;
        }

        public override IDistribution Evaluate(double[] observations)
        {
            throw new System.NotImplementedException();
        }

        public override IDistribution Evaluate(double[][] observations, out double likelihood)
        {
            throw new NotImplementedException("Not multivariate distribution");
        }

        public override IDistribution Evaluate(double[] observations, double[] weights)
        {
            throw new System.NotImplementedException();
        }

        public override IDistribution Evaluate(double[][] observations, double[] weights, out double likelihood)
        {
            throw new NotImplementedException("Not multivariate distribution");
        }

        public IDistribution Evaluate(double[] observations, double[] symbols, double[] gamma)
        {
            var M = symbols.Length;
            var T = observations.Length; 
            var probabilities = new double[M];
            for (var k = 0; k < M; k++)
            {
                double den = 0, num = 0;
                for (var t = 0; t < T; t++)
                {
                    if (observations[t] == symbols[k])
                    {
                        num = num + gamma[t];
                    }
                    den = den + gamma[t];
                }
                probabilities[k] = 1e-10;
                if (num.EqualsToZero())
                {
                    probabilities[k] = den.EqualsToZero() ? 0.0 : LogExtention.eExp(LogExtention.eLnProduct(num, -den));
                }
            }
            return new DiscreteDistributionNormalized(symbols, probabilities);
        }

        public override object Clone()
        {
            return new DiscreteDistribution(_symbols, _probabilities);
        }
    }
}
