using System;
using System.Diagnostics;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.Base;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters;
using HmmDotNet.Mathematic;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation
{
    public class SigmaMultivariateEstimator<TDistribution> : IVariableEstimator<double[][,], SigmaEstimationParameters<TDistribution, double[][]>>
                                                             where TDistribution : IDistribution
    {
        private double[][,] _sigmaMultivariate;

        public double[][,] Estimate(SigmaEstimationParameters<TDistribution, double[][]> parameters)
        {
            if (_sigmaMultivariate != null)
            {
                return _sigmaMultivariate;
            }
            try
            {
                _sigmaMultivariate = new double[parameters.Model.N][,];
                var K = parameters.Observations[0].Dimention;
                var T = parameters.Observations.Count;

                for (var n = 0; n < parameters.Model.N; n++)
                {
                    var covariance = new double[K, K];
                    var nK = 0d;
                    for (var t = 0; t < T; t++)
                    {
                        var x = parameters.Observations[t].Value;
                        var z = x.Substruct(parameters.Mean[n]);
                        var m = z.OuterProduct(z);
                        if (parameters.Model.Normalized)
                        {
                            nK += LogExtention.eExp(parameters.Gamma[t][n]);
                            m = m.Product(LogExtention.eExp(parameters.Gamma[t][n]));
                        }
                        else
                        {
                            nK += parameters.Gamma[t][n];
                            m = m.Product(parameters.Gamma[t][n]);
                        }

                        covariance = covariance.Add(m);
                    }
                    _sigmaMultivariate[n] = covariance.Product(1 / nK);
                    var matrix = new Matrix(_sigmaMultivariate[n]);
                    if (!matrix.PositiviDefinite)
                    {
                        _sigmaMultivariate[n] = matrix.ConvertToPositiveDefinite();
                        Debug.WriteLine("HMM State {0} Sigma is not Positive Definite. Converting.", n);
                        Debug.WriteLine("{0}", matrix);
                    }
                    Debug.WriteLine("HMM State {0} Sigma : {1}", n, new Matrix(_sigmaMultivariate[n]));
                }
            }
            catch (Exception)
            {
                for (var n = 0; n < parameters.Model.N; n++)
                {
                    Debug.WriteLine("HMM State {0} Sigma : {1}", n, new Matrix(_sigmaMultivariate[n]));
                }
                throw;
            }

            return _sigmaMultivariate;
        }
    }
}
