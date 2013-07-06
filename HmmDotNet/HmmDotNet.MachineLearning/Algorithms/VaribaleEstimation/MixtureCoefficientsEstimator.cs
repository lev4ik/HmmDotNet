using System;
using System.Diagnostics;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.Base;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters;
using HmmDotNet.Mathematic;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class MixtureCoefficientsEstimator<TDistribution> : INormalizedEstimator
        where TDistribution : IDistribution
    {
        public MixtureCoefficientsEstimator(IParameterEstimations<TDistribution> parameters)
        {
            _parameters = parameters;
            _gammaComponentsEstimator = new MixtureGammaEstimator<TDistribution>();
            _denormalized = false;
        }

        private bool _denormalized;

        private readonly IParameterEstimations<TDistribution> _parameters;

        private double[][] _coefficients;

        private readonly MixtureGammaEstimator<TDistribution> _gammaComponentsEstimator;

        private double[] _denominator;

        protected double[] Denominator
        {
            get
            {
                if (_denominator == null)
                {
                    var @params = new MixtureAdvancedEstimationParameters<TDistribution>
                    {
                        Alpha = _parameters.Alpha,
                        Beta = _parameters.Beta,
                        L = _parameters.L,
                        Model = _parameters.Model,
                        Normalized = _parameters.Model.Normalized,
                        Observations = _parameters.Observation
                    };
                    _denominator = new double[_parameters.Model.N];
                    for (var i = 0; i < _parameters.Model.N; i++)
                    {
                        _denominator[i] = (_parameters.Model.Normalized) ? double.NaN : 0d;
                        for (var t = 0; t < _parameters.Observation.Count; t++)
                        {
                            if (_parameters.Model.Normalized)
                            {
                                _denominator[i] = LogExtention.eLnSum(_gammaComponentsEstimator.Estimate(@params as AdvancedEstimationParameters<TDistribution>)[t][i],
                                                                      _denominator[i]);
                            }
                            else
                            {
                                _denominator[i] += _gammaComponentsEstimator.Estimate(@params as AdvancedEstimationParameters<TDistribution>)[t][i];
                            }
                        }
                    }
                }
                return _denominator;
            }
        }

        public double[][] Coefficients
        {
            get
            {
                if (_coefficients == null)
                {
                    try
                    {
                        var @params = new MixtureAdvancedEstimationParameters<TDistribution>
                        {
                            Alpha = _parameters.Alpha,
                            Beta = _parameters.Beta,
                            L = _parameters.L,
                            Model = _parameters.Model,
                            Normalized = _parameters.Model.Normalized,
                            Observations = _parameters.Observation
                        };
                        _coefficients = new double[_parameters.Model.N][];
                        for (var i = 0; i < _parameters.Model.N; i++)
                        {
                            _coefficients[i] = new double[_parameters.L];
                            for (var l = 0; l < _parameters.L; l++)
                            {
                                var nominator = (_parameters.Model.Normalized) ? double.NaN : 0d;
                                for (var t = 0; t < _parameters.Observation.Count; t++)
                                {
                                    if (_parameters.Model.Normalized)
                                    {
                                        nominator = LogExtention.eLnSum(_gammaComponentsEstimator.Estimate(@params)[t][i, l],
                                                                        nominator);
                                    }
                                    else
                                    {
                                        nominator += _gammaComponentsEstimator.Estimate(@params)[t][i, l];
                                    }
                                }

                                if (_parameters.Model.Normalized)
                                {
                                    _coefficients[i][l] = LogExtention.eLnProduct(nominator, -Denominator[i]);
                                }
                                else
                                {
                                    _coefficients[i][l] = nominator / Denominator[i];
                                }

                                if (Math.Round(_coefficients[i].Sum(), 5) > 1)
                                {
                                    Debug.WriteLine("Mixture Coeffiecients [{0},{1}] : {2}", i, l, new Vector(_coefficients[i]));
                                    throw new ApplicationException(string.Format("Mixture Coeffiecients is greater than 1. [{0},{1}] : {2} : {3}", i, l, new Vector(_coefficients[i]), Math.Round(_coefficients[i].Sum(), 5)));
                                }
                            }
                        }
                    }
                    catch (Exception)
                    {
                        for (var i = 0; i < _parameters.Model.N; i++)
                        {
                            for (var l = 0; l < _parameters.L; l++)
                            {
                                Debug.WriteLine("Coeffiecients[{0}][{1}] : {2}", i, l, _coefficients[i][l]);
                                if (Math.Round(_coefficients[i].Sum(), 5) > 1)
                                {
                                    Debug.WriteLine("Mixture Coeffiecients [{0},{1}] : {2}", i, l, new Vector(_coefficients[i]));
                                    throw new ApplicationException(string.Format("Mixture Coeffiecients is greater than 1. [{0},{1}] : {2} : {3}", i, l, new Vector(_coefficients[i]), Math.Round(_coefficients[i].Sum(), 5)));
                                }
                            }
                        }                        
                        throw;
                    }
                }
                return _coefficients;
            }
        }

        public void Denormalize()
        {
            if (!_denormalized)
            {
                if (_coefficients == null)
                {
                    var c = Coefficients;
                }
                for (var i = 0; i < _parameters.Model.N; i++)
                {
                    for (var l = 0; l < _parameters.L; l++)
                    {
                        _coefficients[i][l] = LogExtention.eExp(_coefficients[i][l]);
                    }
                }
                _denormalized = true;
            }
        }
    }
}
