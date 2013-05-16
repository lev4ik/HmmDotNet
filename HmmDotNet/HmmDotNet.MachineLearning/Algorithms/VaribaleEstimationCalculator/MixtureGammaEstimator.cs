using System;
using System.Diagnostics;
using HmmDotNet.Mathematic;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class MixtureGammaEstimator<TDistribution> where TDistribution : IDistribution
    {
        public MixtureGammaEstimator(IParameterEstimations<TDistribution> parameters)
        {
            _parameters = parameters;
            _gammaEstimator = new GammaEstimator<TDistribution>(parameters, _parameters.Model.Normalized);
        }

        private readonly GammaEstimator<TDistribution> _gammaEstimator;

        private readonly IParameterEstimations<TDistribution> _parameters;

        private double[][,] _gammaComponents;

        public double[][] Gamma
        {
            get { return _gammaEstimator.Gamma; }
        }

        public double[][,] GammaComponents
        {
            get
            {
                if (_gammaComponents == null)
                {
                    try
                    {
                        _gammaComponents = new double[_parameters.Observation.Count][,];
                        for (var t = 0; t < _parameters.Observation.Count; t++)
                        {
                            _gammaComponents[t] = new double[_parameters.Model.N, _parameters.L];
                            for (var i = 0; i < _parameters.Model.N; i++)
                            {
                                var d = _parameters.Model.Emission[i] as Mixture<IMultivariateDistribution>;
                                if (d != null)
                                {
                                    for (var l = 0; l < _parameters.L; l++)
                                    {
                                        //Emmision in our case are Mixture<T>
                                        var p = d.ProbabilityDensityFunction(l, _parameters.Observation[t].Value);
                                        if (_gammaEstimator.LogNormalized)
                                        {
                                            _gammaComponents[t][i, l] = LogExtention.eLnProduct(_gammaEstimator.Gamma[t][i], LogExtention.eLn(p));
                                        }
                                        else
                                        {
                                            _gammaComponents[t][i, l] = _gammaEstimator.Gamma[t][i] * p;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    catch (Exception)
                    {
                        for (var t = 0; t < _parameters.Observation.Count; t++)
                        {
                            if (Math.Round(_gammaComponents[t].Sum(), 5) > 1)
                            {
                                Debug.WriteLine("Mixture Gamma Components [{0}] : {1}", t, new Matrix(_gammaComponents[t]));
                                throw new ApplicationException(string.Format("Mixture Sigma is greater than 1. [{0}] : {1} : {2}", t, new Matrix(_gammaComponents[t]), Math.Round(_gammaComponents[t].Sum(), 5)));
                            }
                        }                        
                        throw;
                    }

                }
                return _gammaComponents;
            }
        }      
    }
}
