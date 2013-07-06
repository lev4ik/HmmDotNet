using System;
using System.Diagnostics;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.Base;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters;
using HmmDotNet.Mathematic;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class MixtureSigmaEstimator<TDistribution> : IVariableEstimator<double[,][,], MixtureSigmaEstimationParameters<TDistribution>>
                                                        where TDistribution : IDistribution
    {
        public double[,][,] Estimate(MixtureSigmaEstimationParameters<TDistribution> parameters)
        {
            if (_sigma != null)
            {
                return _sigma;
            }
            try
            {
            _sigma = new double[parameters.Model.N, parameters.L][,];
            for (var i = 0; i < parameters.Model.N; i++)
            {
                for (var l = 0; l < parameters.L; l++)
                {
                    var denominator = 0.0d;
                    var nominator = new double[parameters.Observations[0].Dimention, parameters.Observations[0].Dimention];
                    for (var t = 0; t < parameters.Observations.Count; t++)
                    {
                        var gammaComponents = (parameters.Model.Normalized) ? LogExtention.eExp(parameters.GammaComponents[t][i, l]) : parameters.GammaComponents[t][i, l];

                        var x = parameters.Observations[t].Value;
                        var z = x.Substruct(parameters.Mu[i, l]);
                        var m = z.OuterProduct(z);

                        m = m.Product(gammaComponents);
                        denominator += gammaComponents;
                        nominator = nominator.Add(m);
                    }
                    _sigma[i, l] = nominator.Product(1 / denominator);
                    var matrix = new Matrix(_sigma[i, l]);
                    if (!matrix.PositiviDefinite)
                    {
                        _sigma[i, l] = matrix.ConvertToPositiveDefinite();
                        Debug.WriteLine("HMM State [{0},{1}] Sigma is not Positive Definite. Converting.", i, l);
                        Debug.WriteLine("{0}", matrix);
                    }
                }
            }
        }
        catch (Exception)
        {
            for (var i = 0; i < parameters.Model.N; i++)
            {
                for (var l = 0; l < parameters.L; l++)
                {
                    Debug.WriteLine("Mixture Sigma [{0},{1}] : {2}", i, l, new Matrix(_sigma[i, l]));
                }
            }                        
            throw;
        }

            return _sigma;
        }

        private double[,][,] _sigma;
    }
}
