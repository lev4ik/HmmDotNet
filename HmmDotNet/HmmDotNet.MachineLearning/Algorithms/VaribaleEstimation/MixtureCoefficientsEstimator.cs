using System;
using System.Diagnostics;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.Base;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters;
using HmmDotNet.Mathematic;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class MixtureCoefficientsEstimator<TDistribution> : IVariableEstimator<double[][], MixtureCoefficientEstimationParameters<TDistribution>>, 
                                                               INormalizedEstimator
                                                               where TDistribution : IDistribution
    {
        #region Private Methods

        private double GetWeightValue(int t, decimal[] weights)
        {
            var weight = 0.0;
            if (weights != null)
            {
                weight = (_parameters.Model.Normalized) ? LogExtention.eLn((double)weights[t]) : (double)weights[t];
            }
            else
            {
                weight = (_parameters.Model.Normalized) ? 0 : 1;
            }
            return weight;
        }

        #endregion Private Methods

        public double[][] Estimate(MixtureCoefficientEstimationParameters<TDistribution> parameters)
        {
            _parameters = parameters;
            if (_coefficients != null)
            {
                return _coefficients;
            }

            try
            {
                _coefficients = new double[parameters.Model.N][];
                _denominator = new double[parameters.Model.N];
                for (var i = 0; i < parameters.Model.N; i++)
                {
                    _denominator[i] = (parameters.Model.Normalized) ? double.NaN : 0d;
                    for (var t = 0; t < parameters.Observations.Count; t++)
                    {                        
                        var weight = GetWeightValue(t, parameters.ObservationWeights);
                        if (parameters.Normalized)
                        {
                            _denominator[i] = LogExtention.eLnSum(LogExtention.eLnProduct(weight, parameters.Gamma[t][i]), _denominator[i]);
                        }
                        else
                        {
                            _denominator[i] += weight * parameters.Gamma[t][i];
                        }
                    }
                }
                for (var i = 0; i < parameters.Model.N; i++)
                {
                    _coefficients[i] = new double[parameters.L];
                    for (var l = 0; l < parameters.L; l++)
                    {
                        var nominator = (parameters.Model.Normalized) ? double.NaN : 0d;
                        for (var t = 0; t < parameters.Observations.Count; t++)
                        {
                            var weight = GetWeightValue(t, parameters.ObservationWeights);
                            if (parameters.Normalized)
                            {
                                nominator = LogExtention.eLnSum(LogExtention.eLnProduct(weight, parameters.GammaComponents[t][i, l]), nominator);
                            }
                            else
                            {
                                nominator += weight * parameters.GammaComponents[t][i, l];
                            }
                        }

                        if (parameters.Normalized)
                        {
                            _coefficients[i][l] = LogExtention.eLnProduct(nominator, -_denominator[i]);
                        }
                        else
                        {
                            _coefficients[i][l] = nominator / _denominator[i];
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
                for (var i = 0; i < parameters.Model.N; i++)
                {
                    for (var l = 0; l < parameters.L; l++)
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

            return _coefficients;
        }

        private bool _denormalized;

        private double[][] _coefficients;

        private double[] _denominator;
        
        private MixtureCoefficientEstimationParameters<TDistribution> _parameters;

        public void Denormalize()
        {
            if (!_denormalized)
            {
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
