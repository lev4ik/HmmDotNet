using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.Base;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class AlphaEstimator<TDistribution> : IVariableEstimator<double[][], BasicEstimationParameters<TDistribution>> 
                                                 where TDistribution : IDistribution
    {
        private double[][] _alpha;

        public double[][] Estimate(BasicEstimationParameters<TDistribution> parameters)
        {
            if (_alpha != null)
            {
                return _alpha;
            }

            _alpha = new double[parameters.Observations.Count][];
            _alpha[0] = new double[parameters.Model.N];
            // Initialize 
            for (var i = 0; i < parameters.Model.N; i++)
            {
                var o = EstimatorUtilities.GetProbability(parameters.Model.Emission[i], parameters.Observations, 0);
                _alpha[0][i] = (parameters.Normalized) ? LogExtention.eLnProduct(LogExtention.eLn(parameters.Model.Pi[i]), LogExtention.eLn(o)) : parameters.Model.Pi[i] * o;
            }
            // Induction
            for (var t = 1; t < parameters.Observations.Count; t++)
            {
                _alpha[t] = new double[parameters.Model.N];
                for (var j = 0; j < parameters.Model.N; j++)
                {
                    var sum = (parameters.Normalized) ? double.NaN : 0d;
                    for (var i = 0; i < parameters.Model.N; i++)
                    {
                        if (parameters.Normalized)
                        {
                            sum = LogExtention.eLnSum(sum, LogExtention.eLnProduct(_alpha[t - 1][i], LogExtention.eLn(parameters.Model.TransitionProbabilityMatrix[i][j])));
                        }
                        else
                        {
                            sum += _alpha[t - 1][i] * parameters.Model.TransitionProbabilityMatrix[i][j];
                        }
                    }
                    var o = EstimatorUtilities.GetProbability(parameters.Model.Emission[j], parameters.Observations, t);
                    if (parameters.Normalized)
                    {
                        _alpha[t][j] = LogExtention.eLnProduct(sum, LogExtention.eLn(o));
                    }
                    else
                    {
                        _alpha[t][j] = sum * o;
                    }
                }
            }

            return _alpha;
        }
    }
}