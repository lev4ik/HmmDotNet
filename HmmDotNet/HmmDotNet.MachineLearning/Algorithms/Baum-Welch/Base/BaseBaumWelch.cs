using System;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public abstract class BaseBaumWelch<TDistribution> where TDistribution : IDistribution
    {
        #region Variables

        protected readonly double[] _estimatedPi;
        protected readonly double[][] _estimatedTransitionProbabilityMatrix;
        protected readonly IHiddenMarkovModel<TDistribution> _model;
        protected double _likelihoodDelta = 0d;

        #endregion Variables

        #region Constructors

        protected BaseBaumWelch(IHiddenMarkovModel<TDistribution> model)
        {
            _model = model;
            _estimatedPi = new double[_model.N];
            _estimatedTransitionProbabilityMatrix = new double[_model.N][];
        }

        #endregion Constructors

        #region Private Methods

        private void CheckPi(double checksum)
        {
            if (_model.Normalized)
            {
                if (checksum.EqualsToZero() || double.IsNaN(checksum))
                {
                    throw new ApplicationException(string.Format("Pi (LogNormilized) value {0} must be equal to 0", checksum));    
                }
            }
            if (checksum.EqualsToZero() || double.IsNaN(checksum))
            {
                throw new ApplicationException(string.Format("Pi value {0} must be equal to 1", checksum));    
            }
        }

        private void CheckTransitionProbabilityMatrix(double[] checksum)
        {
            for (var i = 0; i < checksum.Length; i++)
            {
                if (_model.Normalized)
                {
                    if (checksum[i].EqualsToZero() || double.IsNaN(checksum[i]))
                    {
                        throw new ApplicationException(string.Format("Transition Probability Matrix (LogNormilized) value {0} must be equal to 0 for element {1}", checksum[i], i));
                    }
                }
                else
                {
                    if (checksum[i].EqualsToZero() || double.IsNaN(checksum[i]))
                    {
                        throw new ApplicationException(string.Format("Transition Probability Matrix value {0} must be equal to 1 for element {1}", checksum[i], i));
                    }                    
                }
            }
        }

        private double GetWeightValue(int t, decimal[] weights)
        {
            var weight = 0.0;
            if (weights != null)
            {
                weight = (_model.Normalized) ? LogExtention.eLn((double)weights[t]) : (double)weights[t];
            }
            else
            {
                weight = (_model.Normalized) ? 0 : 1;
            }
            return weight;
        }

        #endregion Private Methods

        protected void EstimatePi(double[][] gamma)
        {
            var checksum = 0d;
            for (var i = 0; i < _model.N; i++)
            {
                _estimatedPi[i] = (_model.Normalized) ? LogExtention.eExp(gamma[0][i]) : gamma[0][i];
                checksum += _estimatedPi[i];
            }
            CheckPi(checksum);
        }

        protected void EstimateTransitionProbabilityMatrix(double[][] gamma, double[][,] ksi, decimal[] observationWeights, int sequenceLength)
        {
            var checksum = new double[_model.N];
            for (var i = 0; i < _model.N; i++)
            {
                _estimatedTransitionProbabilityMatrix[i] = new double[_model.N];
                for (var j = 0; j < _model.N; j++)
                {
                    double den = (_model.Normalized) ? double.NaN : 0, num = (_model.Normalized) ? double.NaN : 0;
                    for (var t = 0; t < sequenceLength - 1; t++)
                    {
                        var weight = GetWeightValue(t, observationWeights);
                        if (_model.Normalized)
                        {
                            num = LogExtention.eLnSum(num, LogExtention.eLnProduct(weight, ksi[t][i, j]));
                            den = LogExtention.eLnSum(den, LogExtention.eLnProduct(weight, gamma[t][i]));
                        }
                        else
                        {
                            num += weight * ksi[t][i, j];
                            den += weight * gamma[t][i];
                        }
                    }
                    if (_model.Normalized)
                    {
                        _estimatedTransitionProbabilityMatrix[i][j] = den.EqualsToZero() ? 0.0 : LogExtention.eExp(LogExtention.eLnProduct(num, -den));
                    }
                    else
                    {
                        _estimatedTransitionProbabilityMatrix[i][j] = den.EqualsToZero() ? 0.0 : num / den;
                    }
                    checksum[i] = checksum[i] + _estimatedTransitionProbabilityMatrix[i][j];
                }
            }

            CheckTransitionProbabilityMatrix(checksum);
        }
    }
}
