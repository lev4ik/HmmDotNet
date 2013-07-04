using System;
using System.Diagnostics;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator.Base;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator.EstimationParameters;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    /// <summary>
    ///     Calculates Gamma value for all t from 1 to T and for all i from 1 to N
    /// </summary>
    public class GammaEstimator<TDistribution> : IVariableEstimator<double[][], AdvancedEstimationParameters<TDistribution>> 
                                                 where TDistribution : IDistribution
    {
        public double[][] Estimate(AdvancedEstimationParameters<TDistribution> parameters)
        {
            if (_gamma != null)
            {
                return _gamma;
            }

            var denominator = new double[parameters.Observations.Count];
            for (var t = 0; t < parameters.Observations.Count; t++)
            {
                denominator[t] = (parameters.Normalized) ? double.NaN : 0d;
                for (var i = 0; i < parameters.Model.N; i++)
                {
                    if (parameters.Normalized)
                    {
                        denominator[t] = LogExtention.eLnSum(denominator[t], LogExtention.eLnProduct(parameters.Alpha[t][i], parameters.Beta[t][i]));
                    }
                    else
                    {
                        denominator[t] += parameters.Alpha[t][i] * parameters.Beta[t][i];
                    }
                }
            }


            try
            {
                _gamma = new double[parameters.Observations.Count][];
                for (var t = 0; t < parameters.Observations.Count; t++)
                {
                    _gamma[t] = new double[parameters.Model.N];
                    for (var i = 0; i < parameters.Model.N; i++)
                    {
                        if (parameters.Normalized)
                        {
                            _gamma[t][i] = LogExtention.eLnProduct(LogExtention.eLnProduct(parameters.Alpha[t][i], parameters.Beta[t][i]), -denominator[t]);
                        }
                        else
                        {
                            _gamma[t][i] = (parameters.Alpha[t][i] * parameters.Beta[t][i]) / denominator[t];
                        }
                    }
                }
            }
            catch (Exception)
            {
                for (var t = 0; t < parameters.Observations.Count; t++)
                {
                    for (var i = 0; i < parameters.Model.N; i++)
                    {
                        Debug.WriteLine("Gamma [{0}][{1}] : alpha : {2} , beta : {3} , denominator : {4} : gamma {5} ", t, i, parameters.Alpha[t][i], parameters.Beta[t][i], denominator[t], _gamma[t][i]);
                    }
                }
                throw;
            }

            return _gamma;
        }

        private double[][] _gamma;
    }
}
