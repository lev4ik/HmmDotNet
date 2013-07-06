using System;
using System.Diagnostics;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.Base;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class BetaEstimator<TDistribution> : IVariableEstimator<double[][], BasicEstimationParameters<TDistribution>>
                                                where TDistribution : IDistribution
    {
        private double[][] _beta;

        public double[][] Estimate(BasicEstimationParameters<TDistribution> parameters)
        {
            if (_beta != null)
            {
                return _beta;
            }
            var T = parameters.Observations.Count;
            try
            {
                _beta = new double[T][];
                _beta[T - 1] = new double[parameters.Model.N];

                for (var i = 0; i < parameters.Model.N; i++)
                {
                    _beta[T - 1][i] = (parameters.Normalized) ? 0d : 1d;
                }

                for (var t = T - 2; t >= 0; t--)
                {
                    _beta[t] = new double[parameters.Model.N];
                    for (var i = 0; i < parameters.Model.N; i++)
                    {
                        _beta[t][i] = (parameters.Normalized) ? double.NaN : 0d;
                        for (var j = 0; j < parameters.Model.N; j++)
                        {
                            var o = EstimatorUtilities.GetProbability(parameters.Model.Emission[j], parameters.Observations, t + 1);
                            if (parameters.Normalized)
                            {
                                _beta[t][i] = LogExtention.eLnSum(_beta[t][i],
                                                                   LogExtention.eLnProduct(LogExtention.eLn(parameters.Model.TransitionProbabilityMatrix[i][j]),
                                                                                           LogExtention.eLnProduct(LogExtention.eLn(o), _beta[t + 1][j])));
                            }
                            else
                            {
                                _beta[t][i] += parameters.Model.TransitionProbabilityMatrix[i][j] * o * _beta[t + 1][j];
                            }
                        }
                    }
                }
            }
            catch (Exception)
            {
                for (var t = T - 2; t >= 0; t--)
                {
                    for (var i = 0; i < parameters.Model.N; i++)
                    {
                        for (var j = 0; j < parameters.Model.N; j++)
                        {
                            Debug.WriteLine("[{0}][{1}] : beta : {2}", t, i, _beta[t][i]);
                        }
                    }
                }
                throw;
            }

            return _beta;
        }
    }
}
