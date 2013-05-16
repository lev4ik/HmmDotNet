using System;
using System.Collections.Generic;
using System.Diagnostics;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class BetaEstimator<T> : BaseEstimator where T : IDistribution
    {
        private double[][] _beta;

        public BetaEstimator(IHiddenMarkovModelState<T> model, IList<IObservation> observations, bool logNormalized)
        {
            LogNormalized = logNormalized;
            _model = model;
            _observations = observations;
        }

        private readonly IHiddenMarkovModelState<T> _model;
        private readonly IList<IObservation> _observations;


        public double[][] Beta
        {
            get
            {
                if (_beta == null)
                {
                    var T = _observations.Count;
                    try
                    {                        
                        _beta = new double[T][];
                        _beta[T - 1] = new double[_model.N];

                        for (var i = 0; i < _model.N; i++)
                        {
                            _beta[T - 1][i] = (LogNormalized) ? 0d : 1d;
                        }

                        for (var t = T - 2; t >= 0; t--)
                        {
                            _beta[t] = new double[_model.N];
                            for (var i = 0; i < _model.N; i++)
                            {
                                _beta[t][i] = (LogNormalized) ? double.NaN : 0d;
                                for (var j = 0; j < _model.N; j++)
                                {
                                    var o = GetProbability(_model.Emission[j], _observations, t + 1);
                                    if (LogNormalized)
                                    {
                                        _beta[t][i] = LogExtention.eLnSum(_beta[t][i],
                                                                           LogExtention.eLnProduct(LogExtention.eLn(_model.TransitionProbabilityMatrix[i][j]),
                                                                                                   LogExtention.eLnProduct(LogExtention.eLn(o), _beta[t + 1][j])));
                                    }
                                    else
                                    {
                                        _beta[t][i] += _model.TransitionProbabilityMatrix[i][j] * o * _beta[t + 1][j];
                                    }
                                }
                            }
                        }
                    }
                    catch (Exception)
                    {
                        for (var t = T - 2; t >= 0; t--)
                        {
                            for (var i = 0; i < _model.N; i++)
                            {
                                for (var j = 0; j < _model.N; j++)
                                {
                                    Debug.WriteLine("[{0}][{1}] : beta : {2}", t, i, _beta[t][i]);
                                }
                            }
                        }                        
                        throw;
                    }

                }
                return _beta;
            }
        }
    }
}
