using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.Base;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class MuUnivariateEstimator<TDistribution> : IVariableEstimator<double[], MuEstimationParameters<TDistribution>>
                                              where TDistribution : IDistribution
    {
        public double[] Estimate(MuEstimationParameters<TDistribution> parameters)
        {
            if (_muUnivariate != null)
            {
                return _muUnivariate;
            }
            _muUnivariate = new double[parameters.Model.N];
            for (var n = 0; n < parameters.Model.N; n++)
            {
                var T = parameters.Observations.Count;
                var mean = 0d;
                var nK = 0d;
                for (var t = 0; t < T; t++)
                {
                    if (parameters.Model.Normalized)
                    {
                        nK += LogExtention.eExp(parameters.Gamma[t][n]);
                        mean += parameters.Observations[t].Value[0] * LogExtention.eExp(parameters.Gamma[t][n]);
                    }
                    else
                    {
                        nK += parameters.Gamma[t][n];
                        mean += parameters.Observations[t].Value[0] * parameters.Gamma[t][n];
                    }
                }

                _muUnivariate[n] = mean / nK;
            }

            return _muUnivariate;
        }

        private double[] _muUnivariate;
    }
}
