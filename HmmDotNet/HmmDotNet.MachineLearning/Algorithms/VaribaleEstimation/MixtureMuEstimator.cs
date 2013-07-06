using System;
using System.Diagnostics;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.Base;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters;
using HmmDotNet.Mathematic;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class MixtureMuEstimator<TDistribution> : IVariableEstimator<double[,][], MixtureCoefficientEstimationParameters<TDistribution>> 
                                                     where TDistribution : IDistribution
    {
        public double[,][] Estimate(MixtureCoefficientEstimationParameters<TDistribution> parameters)
        {
            if (_mu != null)
            {
                return _mu;
            }

            try
            {
                _mu = new double[parameters.Model.N, parameters.L][];
                for (var i = 0; i < parameters.Model.N; i++)
                {
                    for (var l = 0; l < parameters.L; l++)
                    {
                        var denominator = 0.0d;
                        var nominator = new double[parameters.Observations[0].Dimention];
                        for (var t = 0; t < parameters.Observations.Count; t++)
                        {
                            var x = parameters.Observations[t].Value;
                            var gamma = (parameters.Model.Normalized) ? LogExtention.eExp(parameters.GammaComponents[t][i, l])
                                                                          : parameters.GammaComponents[t][i, l];
                            denominator += gamma;
                            x = x.Product(gamma);
                            nominator = nominator.Add(x);
                        }
                        _mu[i, l] = nominator.Product(1 / denominator);
                    }
                }
            }
            catch (Exception)
            {
                for (var i = 0; i < parameters.Model.N; i++)
                {
                    for (var l = 0; l < parameters.L; l++)
                    {
                        Debug.WriteLine("Mixture Mu [{0},{1}] : {2}", i, l, new Vector(_mu[i, l]));
                    }
                }
                throw;
            }

            return _mu;
        }

        private double[,][] _mu;
    }
}
