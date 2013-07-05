using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator.Base;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator.EstimationParameters;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    /// <summary>
    ///     Calculates Ksi value for all t from 1 to T and for all i from 1 to N
    /// </summary>
    public class KsiEstimator<TDistribution> : BaseEstimator, IVariableEstimator<double[][,], AdvancedEstimationParameters<TDistribution>> where TDistribution : IDistribution
    {
        public double[][,] Estimate(AdvancedEstimationParameters<TDistribution> parameters)
        {
            if (_ksi != null)
            {
                return _ksi;
            }
            var denominator = new double[parameters.Observations.Count];
            for (var t = 0; t < parameters.Observations.Count - 1; t++)
            {
                denominator[t] = (parameters.Normalized) ? double.NaN : 0d;
                for (var i = 0; i < parameters.Model.N; i++)
                {
                    for (var j = 0; j < parameters.Model.N; j++)
                    {
                        var o = EstimatorUtilities.GetProbability(parameters.Model.Emission[j], parameters.Observations, t + 1);
                        if (parameters.Normalized)
                        {
                            denominator[t] = LogExtention.eLnSum(denominator[t], LogExtention.eLnProduct(parameters.Alpha[t][i],
                                                                                                         LogExtention.eLnProduct(LogExtention.eLn(parameters.Model.TransitionProbabilityMatrix[i][j]),
                                                                                                         LogExtention.eLnProduct(parameters.Beta[t + 1][j], LogExtention.eLn(o)))));
                        }
                        else
                        {
                            denominator[t] += parameters.Alpha[t][i] * parameters.Model.TransitionProbabilityMatrix[i][j] * parameters.Beta[t + 1][j] * o;
                        }
                    }
                }
            }

            _ksi = new double[parameters.Observations.Count][,];
            for (var t = 0; t < parameters.Observations.Count - 1; t++)
            {
                _ksi[t] = new double[parameters.Model.N, parameters.Model.N];
                for (var i = 0; i < parameters.Model.N; i++)
                {
                    for (var j = 0; j < parameters.Model.N; j++)
                    {
                        var o = EstimatorUtilities.GetProbability(parameters.Model.Emission[j], parameters.Observations, t + 1);
                        if (parameters.Normalized)
                        {
                            var nominator = LogExtention.eLnProduct(parameters.Alpha[t][i],
                                                          LogExtention.eLnProduct(LogExtention.eLn(parameters.Model.TransitionProbabilityMatrix[i][j]),
                                                          LogExtention.eLnProduct(parameters.Beta[t + 1][j], LogExtention.eLn(o))));
                            _ksi[t][i, j] = LogExtention.eLnProduct(nominator, -denominator[t]);
                        }
                        else
                        {
                            var nominator = parameters.Alpha[t][i] * parameters.Model.TransitionProbabilityMatrix[i][j] * parameters.Beta[t + 1][j] * o;
                            _ksi[t][i, j] = nominator / denominator[t];
                        }
                    }
                }
            }

            return _ksi;
        }

        private double[][,] _ksi;
    }
}
