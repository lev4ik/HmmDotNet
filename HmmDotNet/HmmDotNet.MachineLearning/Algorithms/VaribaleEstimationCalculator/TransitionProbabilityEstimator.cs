using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator.Base;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator
{
    public class TransitionProbabilityEstimator<TDistribution> : IVariableEstimator<double[][]> where TDistribution : IDistribution
    {
        private readonly double[][] _alpha;
        private readonly double[][] _beta;
        private readonly double[][] _transitionProbabilityMatrix;
        private readonly TDistribution[] _emission;
        private readonly double[][] _observations;
        private readonly double[] _weights;

        private double[][] _estimatedTransitionProbabilityMatrix;

        public TransitionProbabilityEstimator(double[][] alpha, double[][] beta, double[][] transitionProbabilityMatrix, TDistribution[] emission, double[][] observations, double[] weights)
        {
            _alpha = alpha;
            _beta = beta; 
            _transitionProbabilityMatrix = transitionProbabilityMatrix;
            _emission = emission;
            _observations = observations;
            _weights = weights;
        }

        public double[][] Estimate(bool normalized)
        {
            if (_estimatedTransitionProbabilityMatrix != null)
            {
                return _transitionProbabilityMatrix;
            }
            var T = _observations.Length;
            var N = _emission.Length;

            _estimatedTransitionProbabilityMatrix = new double[N][];

            for (var i = 0; i < N; i++)
            {
                _estimatedTransitionProbabilityMatrix[i] = new double[N];
                for (var j = 0; j < N; j++)
                {
                    var nominator = (normalized) ? double.NaN : 0.0d;
                    var denominator = (normalized) ? double.NaN : 0.0d;
                    for (var t = 0; t < T; t++)
                    {
                        nominator = CalculateNominatorForTimet(nominator, i, j, t, normalized);
                        denominator = CalculateDenominatorForTimet(denominator, i, j, t, normalized);
                    }
                    _estimatedTransitionProbabilityMatrix[i][j] = nominator / denominator;
                }
            }
            return _estimatedTransitionProbabilityMatrix;
        }

        private double CalculateNominatorForTimet(double nominator, int i, int j, int t, bool normalized)
        {
            return (normalized)
                       ? LogExtention.eLnSum(nominator, LogExtention.eLnProduct(_weights[t], LogExtention.eLnProduct(_alpha[i][t], LogExtention.eLnProduct(_transitionProbabilityMatrix[i][j], LogExtention.eLnProduct(_emission[i].ProbabilityDensityFunction(_observations[t]), _beta[j][t])))))
                       : nominator + _weights[t] * _alpha[i][t] * _transitionProbabilityMatrix[i][j] * _emission[i].ProbabilityDensityFunction(_observations[t]) * _beta[j][t];
        }

        private double CalculateDenominatorForTimet(double denominator, int i, int j, int t, bool normalized)
        {
            return (normalized) ? LogExtention.eLnSum(denominator, LogExtention.eLnProduct(_weights[t], LogExtention.eLnProduct(_alpha[i][t] , _beta[j][t]))) : denominator + _weights[t] * _alpha[i][t] * _beta[j][t];
        }
    }
}
