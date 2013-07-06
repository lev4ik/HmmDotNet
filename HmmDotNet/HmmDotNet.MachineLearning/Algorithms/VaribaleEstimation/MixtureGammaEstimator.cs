using System;
using System.Diagnostics;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.Base;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters;
using HmmDotNet.Mathematic;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class MixtureGammaEstimator<TDistribution> : IVariableEstimator<double[][,], MixtureAdvancedEstimationParameters<TDistribution>>,
                                                        IVariableEstimator<double[][], AdvancedEstimationParameters<TDistribution>>  
                                                        where TDistribution : IDistribution
    {
        public double[][,] Estimate(MixtureAdvancedEstimationParameters<TDistribution> parameters)
        {
            if (_gammaComponents != null)
            {
                return _gammaComponents;
            }

            try
            {
                _gammaComponents = new double[parameters.Observations.Count][,];
                for (var t = 0; t < parameters.Observations.Count; t++)
                {
                    _gammaComponents[t] = new double[parameters.Model.N, parameters.L];
                    for (var i = 0; i < parameters.Model.N; i++)
                    {
                        var d = parameters.Model.Emission[i] as Mixture<IMultivariateDistribution>;
                        if (d != null)
                        {
                            for (var l = 0; l < parameters.L; l++)
                            {
                                //Emmision in our case are Mixture<T>
                                var p = d.ProbabilityDensityFunction(l, parameters.Observations[t].Value);
                                if (parameters.Normalized)
                                {
                                    _gammaComponents[t][i, l] = LogExtention.eLnProduct(_gammaEstimator.Estimate(parameters)[t][i], LogExtention.eLn(p));
                                }
                                else
                                {
                                    _gammaComponents[t][i, l] = _gammaEstimator.Estimate(parameters)[t][i] * p;
                                }
                            }
                        }
                    }
                }
            }
            catch (Exception)
            {
                for (var t = 0; t < parameters.Observations.Count; t++)
                {
                    if (Math.Round(_gammaComponents[t].Sum(), 5) > 1)
                    {
                        Debug.WriteLine("Mixture Gamma Components [{0}] : {1}", t, new Matrix(_gammaComponents[t]));
                        throw new ApplicationException(string.Format("Mixture Sigma is greater than 1. [{0}] : {1} : {2}", t, new Matrix(_gammaComponents[t]), Math.Round(_gammaComponents[t].Sum(), 5)));
                    }
                }
                throw;
            }


            return _gammaComponents;
        }

        public double[][] Estimate(AdvancedEstimationParameters<TDistribution> parameters)
        {
            return _gammaEstimator.Estimate(parameters);
        }

        private readonly GammaEstimator<TDistribution> _gammaEstimator = new GammaEstimator<TDistribution>();

        private double[][,] _gammaComponents;
    }
}
