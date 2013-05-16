using System;
using System.Diagnostics;
using HmmDotNet.Mathematic;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class MixtureMuEstimator<TDistribution> where TDistribution : IDistribution
    {
        public MixtureMuEstimator(IParameterEstimations<TDistribution> parameters)
        {
            _parameters = parameters;
            _gammaComponentsEstimator = new MixtureGammaEstimator<TDistribution>(parameters);
        }

        private readonly IParameterEstimations<TDistribution> _parameters;

        private readonly MixtureGammaEstimator<TDistribution> _gammaComponentsEstimator;

        private double[,][] _mu;

        public double[,][] Mu
        {
            get
            {
                if(_mu == null)
                {
                    try
                    {
                        _mu = new double[_parameters.Model.N, _parameters.L][];
                        for (var i = 0; i < _parameters.Model.N; i++)
                        {
                            for (var l = 0; l < _parameters.L; l++)
                            {
                                var denominator = 0.0d;
                                var nominator = new double[_parameters.Observation[0].Dimention];
                                for (var t = 0; t < _parameters.Observation.Count; t++)
                                {
                                    var x = _parameters.Observation[t].Value;
                                    var gamma = (_parameters.Model.Normalized) ? LogExtention.eExp(_gammaComponentsEstimator.GammaComponents[t][i, l])
                                                                                  : _gammaComponentsEstimator.GammaComponents[t][i, l];
                                    denominator += gamma;
                                    x = x.Product(gamma);
                                    nominator = nominator.Add(x);
                                }
                                _mu[i, l] = nominator.Product(1 / denominator);
                            }
                        }
                    }
                    catch (Exception)
                    {
                        for (var i = 0; i < _parameters.Model.N; i++)
                        {
                            for (var l = 0; l < _parameters.L; l++)
                            {
                                Debug.WriteLine("Mixture Mu [{0},{1}] : {2}", i, l, new Vector(_mu[i, l]));
                            }
                        }                        
                        throw;
                    }

                }
                return _mu;
            }
        }
    }
}
