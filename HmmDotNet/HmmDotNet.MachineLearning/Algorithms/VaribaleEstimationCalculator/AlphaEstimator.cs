using System.Collections.Generic;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class AlphaEstimator<T> : BaseEstimator where T : IDistribution
    {
        public AlphaEstimator(IHiddenMarkovModelState<T> model, IList<IObservation> observations, bool logNormalized)
        {
            LogNormalized = logNormalized;
            _model = model;
            _observations = observations;
        }

        private readonly IHiddenMarkovModelState<T> _model;
        private readonly IList<IObservation> _observations;

        private double[][] _alpha;

        public double[][] Alpha
        {
            get
            {
                if (_alpha == null)
                {
                    _alpha = new double[_observations.Count][];
                    _alpha[0] = new double[_model.N];
                    // Initialize 
                    for (var i = 0; i < _model.N; i++)
                    {
                        var o = GetProbability(_model.Emission[i], _observations, 0);
                        _alpha[0][i] = (LogNormalized) ? LogExtention.eLnProduct(LogExtention.eLn(_model.Pi[i]), LogExtention.eLn(o)) : _model.Pi[i] * o;
                    }
                    // Induction
                    for (var t = 1; t < _observations.Count; t++)
                    {
                        _alpha[t] = new double[_model.N];
                        for (var j = 0; j < _model.N; j++)
                        {
                            var sum = (LogNormalized) ? double.NaN : 0d;
                            for (var i = 0; i < _model.N; i++)
                            {
                                if (LogNormalized)
                                {
                                    sum = LogExtention.eLnSum(sum, LogExtention.eLnProduct(_alpha[t - 1][i], LogExtention.eLn(_model.TransitionProbabilityMatrix[i][j])));
                                }
                                else
                                {
                                    sum += _alpha[t - 1][i] * _model.TransitionProbabilityMatrix[i][j];
                                }
                            }
                            var o = GetProbability(_model.Emission[j], _observations, t);
                            if (LogNormalized)
                            {
                                _alpha[t][j] = LogExtention.eLnProduct(sum, LogExtention.eLn(o));
                            }
                            else
                            {
                                _alpha[t][j] = sum * o;
                            }
                        }
                    }
                }
                return _alpha;
            }
        }
    }
}