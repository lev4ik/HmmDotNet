using System;
using System.Diagnostics;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.Base;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters;
using HmmDotNet.Mathematic;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class MuMultivariateEstimator<TDistribution> : IVariableEstimator<double[][], MuEstimationParameters<TDistribution>>
                                                          where TDistribution : IDistribution
    {
        private double[][] _muMultivariate;

        public double[][] Estimate(MuEstimationParameters<TDistribution> parameters)
        {
            if (_muMultivariate != null)
            {
                return _muMultivariate;
            }
            try
            {
                _muMultivariate = new double[parameters.Model.N][];
                var K = parameters.Observations[0].Dimention; // Number of dimentions
                var T = parameters.Observations.Count;

                for (var n = 0; n < parameters.Model.N; n++)
                {
                    var mean = new double[K];
                    var nK = 0d;
                    for (var t = 0; t < T; t++)
                    {
                        if (parameters.Model.Normalized)
                        {
                            nK += LogExtention.eExp(parameters.Gamma[t][n]);
                            mean = mean.Add(parameters.Observations[t].Value.Product(LogExtention.eExp(parameters.Gamma[t][n])));
                        }
                        else
                        {
                            nK += parameters.Gamma[t][n];
                            mean = mean.Add(parameters.Observations[t].Value.Product(parameters.Gamma[t][n]));
                        }
                    }

                    _muMultivariate[n] = mean.Product(1 / nK);
                    Debug.WriteLine(string.Format("HMM State {0} : Mu {1}", n, new Vector(_muMultivariate[n])));
                }
            }
            catch (Exception)
            {
                for (var n = 0; n < parameters.Model.N; n++)
                {
                    Debug.WriteLine(string.Format("HMM State {0} : Mu {1}", n, new Vector(_muMultivariate[n])));
                }
                throw;
            }
            return _muMultivariate;
        }
    }
}
