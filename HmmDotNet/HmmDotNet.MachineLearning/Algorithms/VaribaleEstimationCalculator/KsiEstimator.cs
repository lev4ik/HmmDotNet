using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    /// <summary>
    ///     Calculates Ksi value for all t from 1 to T and for all i from 1 to N
    /// </summary>
    public class KsiEstimator<TDistribution> : BaseEstimator where TDistribution : IDistribution
    {
        public KsiEstimator(IParameterEstimations<TDistribution> parameters, bool logNormalized)
        {
            LogNormalized = logNormalized;
            _parameters = parameters;
            CalculateDenominator();
            CalculateKsi();
        }

        private readonly IParameterEstimations<TDistribution> _parameters;

        private double[][,] _ksi;

        private double[] _denominator;

        protected double[] Denominator
        {
            get
            {
                return _denominator;
            }
        }

        public double[][,] Ksi
        {
            get
            {
                return _ksi;
            }
        }

        #region Private Methods

        private void CalculateDenominator()
        {
            if (_denominator == null)
            {
                _denominator = new double[_parameters.Observation.Count];
                for (var t = 0; t < _parameters.Observation.Count - 1; t++)
                {
                    _denominator[t] = (LogNormalized) ? double.NaN : 0d;
                    for (var i = 0; i < _parameters.Model.N; i++)
                    {
                        for (var j = 0; j < _parameters.Model.N; j++)
                        {
                            var o = GetProbability(_parameters.Model.Emission[j], _parameters.Observation, t + 1);
                            if (LogNormalized)
                            {
                                _denominator[t] = LogExtention.eLnSum(_denominator[t], LogExtention.eLnProduct(_parameters.Alpha[t][i],
                                                                                                                 LogExtention.eLnProduct(LogExtention.eLn(_parameters.Model.TransitionProbabilityMatrix[i][j]),
                                                                                                                 LogExtention.eLnProduct(_parameters.Beta[t + 1][j], LogExtention.eLn(o)))));
                            }
                            else
                            {
                                _denominator[t] += _parameters.Alpha[t][i] * _parameters.Model.TransitionProbabilityMatrix[i][j] * _parameters.Beta[t + 1][j] * o;
                            }
                        }
                    }
                }
            }
        }

        private void CalculateKsi()
        {
            if (_ksi == null)
            {
                _ksi = new double[_parameters.Observation.Count][,];
                for (var t = 0; t < _parameters.Observation.Count - 1; t++)
                {
                    _ksi[t] = new double[_parameters.Model.N, _parameters.Model.N];
                    for (var i = 0; i < _parameters.Model.N; i++)
                    {
                        for (var j = 0; j < _parameters.Model.N; j++)
                        {
                            var o = GetProbability(_parameters.Model.Emission[j], _parameters.Observation, t + 1);
                            if (LogNormalized)
                            {
                                var nominator = LogExtention.eLnProduct(_parameters.Alpha[t][i],
                                                              LogExtention.eLnProduct(LogExtention.eLn(_parameters.Model.TransitionProbabilityMatrix[i][j]),
                                                              LogExtention.eLnProduct(_parameters.Beta[t + 1][j], LogExtention.eLn(o))));
                                _ksi[t][i, j] = LogExtention.eLnProduct(nominator, -Denominator[t]);
                            }
                            else
                            {
                                var nominator = _parameters.Alpha[t][i] * _parameters.Model.TransitionProbabilityMatrix[i][j] * _parameters.Beta[t + 1][j] * o;
                                _ksi[t][i, j] = nominator / Denominator[t];
                            }
                        }
                    }
                }
            }
        }
        #endregion Private Methods
    }
}
