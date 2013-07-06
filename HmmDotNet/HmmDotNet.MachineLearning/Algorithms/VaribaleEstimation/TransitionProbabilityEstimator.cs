using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.Base;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class TransitionProbabilityEstimator<TDistribution> : IVariableEstimator<double[][], AlphaBetaTransitionProbabiltyMatrixParameters<TDistribution>>,
                                                                 IVariableEstimator<double[][], KsiGammaTransitionProbabilityMatrixParameters<TDistribution>> 
                                                                 where TDistribution : IDistribution
    {
        private double[][] _estimatedTransitionProbabilityMatrix;

        private double CalculateTransitionProbabilityMatrixEntry(double nominator, double denominator, bool normalized)
        {
            if (denominator.EqualsToZero())
            {
                return 0;
            }
            return (normalized) ? LogExtention.eExp(LogExtention.eLnProduct(nominator, -denominator)) : nominator / denominator;
        }

        public double[][] Estimate(AlphaBetaTransitionProbabiltyMatrixParameters<TDistribution> parameters)
        {
            if (_estimatedTransitionProbabilityMatrix != null)
            {
                return _estimatedTransitionProbabilityMatrix;
            }
            var T = parameters.Observations.Length;

            _estimatedTransitionProbabilityMatrix = new double[parameters.Model.N][];

            for (var i = 0; i < parameters.Model.N; i++)
            {
                _estimatedTransitionProbabilityMatrix[i] = new double[parameters.Model.N];
                for (var j = 0; j < parameters.Model.N; j++)
                {
                    double nominator = (parameters.Normalized) ? double.NaN : 0.0d, denominator = (parameters.Normalized) ? double.NaN : 0.0d;
                    for (var t = 0; t < T - 1; t++)
                    {
                        var probability = parameters.Model.Emission[j].ProbabilityDensityFunction(parameters.Observations[t + 1]);
                        if (parameters.Normalized)
                        {
                            nominator = LogExtention.eLnSum(nominator, LogExtention.eLnProduct(parameters.Weights[t], LogExtention.eLnProduct(parameters.Alpha[t][i], LogExtention.eLnProduct(LogExtention.eLn(parameters.Model.TransitionProbabilityMatrix[i][j]), LogExtention.eLnProduct(LogExtention.eLn(probability), parameters.Beta[t + 1][j])))));
                            denominator = LogExtention.eLnSum(denominator, LogExtention.eLnProduct(parameters.Weights[t], LogExtention.eLnProduct(parameters.Alpha[t][i], parameters.Beta[t][j])));
                        }
                        else
                        {
                            nominator += parameters.Weights[t] * parameters.Alpha[t][i] * parameters.Model.TransitionProbabilityMatrix[i][j] * probability * parameters.Beta[t + 1][j];
                            denominator += parameters.Weights[t] * parameters.Alpha[t][i] * parameters.Beta[t][j];
                        }                        
                    }
                    _estimatedTransitionProbabilityMatrix[i][j] = CalculateTransitionProbabilityMatrixEntry(nominator, denominator, parameters.Normalized);
                }
            }
            return _estimatedTransitionProbabilityMatrix;
        }

        public double[][] Estimate(KsiGammaTransitionProbabilityMatrixParameters<TDistribution> parameters)
        {
            if (_estimatedTransitionProbabilityMatrix != null)
            {
                return _estimatedTransitionProbabilityMatrix;
            }
            _estimatedTransitionProbabilityMatrix = new double[parameters.Model.N][];

            for (var i = 0; i < parameters.Model.N; i++)
            {
                _estimatedTransitionProbabilityMatrix[i] = new double[parameters.Model.N];
                for (var j = 0; j < parameters.Model.N; j++)
                {
                    double denominator = (parameters.Model.Normalized) ? double.NaN : 0, nominator = (parameters.Model.Normalized) ? double.NaN : 0;
                    for (var t = 0; t < parameters.T - 1; t++)
                    {
                        if (parameters.Model.Normalized)
                        {
                            nominator = LogExtention.eLnSum(nominator, parameters.Ksi[t][i, j]);
                            denominator = LogExtention.eLnSum(denominator, parameters.Gamma[t][i]);
                        }
                        else
                        {
                            nominator += parameters.Ksi[t][i, j];
                            denominator += parameters.Gamma[t][i];
                        }
                    }
                    _estimatedTransitionProbabilityMatrix[i][j] = CalculateTransitionProbabilityMatrixEntry(nominator, denominator, parameters.Normalized);
                }
            }
            return _estimatedTransitionProbabilityMatrix;
        }
    }
}
