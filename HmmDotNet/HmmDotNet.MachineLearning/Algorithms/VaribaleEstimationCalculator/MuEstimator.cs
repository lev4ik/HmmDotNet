using System;
using System.Collections.Generic;
using System.Diagnostics;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Mathematic;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class MuEstimator<TDistribution> where TDistribution : IDistribution
    {
        public MuEstimator(IHiddenMarkovModelState<TDistribution> model, IList<IObservation> observations)
        {
            _model = model;
            _observations = observations;
        }

        private readonly IHiddenMarkovModelState<TDistribution> _model;

        private readonly IList<IObservation> _observations;

        private double[] _muUnivariate;

        private double[][] _muMultivariate;
        //BUG : Handle NaN in gamma
        //BUG : Handle LogNormalized calculation
        public double[] MuUnivariate(double[][] gamma)
        {
            if (_muUnivariate == null)
            {
                _muUnivariate = new double[_model.N];
                for (var n = 0; n < _model.N; n++)
                {
                    var T = _observations.Count;
                    var mean = 0d;
                    var nK = 0d;
                    for (var t = 0; t < T; t++)
                    {
                        nK += gamma[t][n];
                        mean = mean + _observations[t].Value[0] * gamma[t][n];
                    }

                    _muUnivariate[n] = mean / nK;
                    
                }
            }
            return _muUnivariate;
        }
        //BUG : Handle NaN in gamma
        //BUG : Handle LogNormalized calculation
        /// <summary>
        ///     Mu[NumberOfComponents][Dimentions]
        /// </summary>
        /// <param name="gamma"></param>
        /// <returns></returns>
        public double[][] MuMultivariate(double[][] gamma)
        {
            if (_muMultivariate == null)
            {
                try
                {
                    _muMultivariate = new double[_model.N][];
                    var K = _observations[0].Dimention; // Number of dimentions
                    var T = _observations.Count;

                    for (var n = 0; n < _model.N; n++)
                    {
                        var mean = new double[K];
                        var nK = 0d;
                        for (var t = 0; t < T; t++)
                        {
                            if (_model.Normalized)
                            {
                                nK += LogExtention.eExp(gamma[t][n]);
                                mean = mean.Add(_observations[t].Value.Product(LogExtention.eExp(gamma[t][n])));
                            }
                            else
                            {
                                nK += gamma[t][n];
                                mean = mean.Add(_observations[t].Value.Product(gamma[t][n]));
                            }
                        }

                        _muMultivariate[n] = mean.Product(1 / nK);
                        Debug.WriteLine(string.Format("HMM State {0} : Mu {1}", n, new Vector(_muMultivariate[n])));
                    }
                }
                catch (Exception)
                {
                    for (var n = 0; n < _model.N; n++)
                    {
                        Debug.WriteLine(string.Format("HMM State {0} : Mu {1}", n, new Vector(_muMultivariate[n])));
                    }                    
                    throw;
                }

            }
            return _muMultivariate;
        }
    }
}
