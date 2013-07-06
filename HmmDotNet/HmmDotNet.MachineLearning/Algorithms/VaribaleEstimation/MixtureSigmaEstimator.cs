using System;
using System.Diagnostics;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters;
using HmmDotNet.Mathematic;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class MixtureSigmaEstimator<TDistribution> where TDistribution : IDistribution
    {
        public MixtureSigmaEstimator(IParameterEstimations<TDistribution> parameters)
        {
            _parameters = parameters;
            _gammaComponentsEstimator = new MixtureGammaEstimator<TDistribution>();
            _muEstimator = new MixtureMuEstimator<TDistribution>(_parameters);
        }

        private readonly IParameterEstimations<TDistribution> _parameters;

        private readonly MixtureGammaEstimator<TDistribution> _gammaComponentsEstimator;

        private readonly MixtureMuEstimator<TDistribution> _muEstimator;

        private double[,][,] _sigma;

        public double[,][,] Sigma
        {
            get
            {
                if (_sigma == null)
                {
                    try
                    {
                        var @params = new MixtureAdvancedEstimationParameters<TDistribution>
                        {
                            Alpha = _parameters.Alpha,
                            Beta = _parameters.Beta,
                            L = _parameters.L,
                            Model = _parameters.Model,
                            Normalized = _parameters.Model.Normalized,
                            Observations = _parameters.Observation
                        };
                        _sigma = new double[_parameters.Model.N, _parameters.L][,];
                        for (var i = 0; i < _parameters.Model.N; i++)
                        {
                            for (var l = 0; l < _parameters.L; l++)
                            {
                                var denominator = 0.0d;
                                var nominator = new double[_parameters.Observation[0].Dimention, _parameters.Observation[0].Dimention];
                                for (var t = 0; t < _parameters.Observation.Count; t++)
                                {
                                    var x = _parameters.Observation[t].Value;
                                    var z = x.Substruct(_muEstimator.Mu[i, l]);
                                    var m = z.OuterProduct(z);
                                    var gamma = (_parameters.Model.Normalized) ? LogExtention.eExp(_gammaComponentsEstimator.Estimate(@params)[t][i, l])
                                                                                  : _gammaComponentsEstimator.Estimate(@params)[t][i, l];
                                    m = m.Product(gamma);
                                    denominator += gamma;
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
                        for (var i = 0; i < _parameters.Model.N; i++)
                        {
                            for (var l = 0; l < _parameters.L; l++)
                            {
                                Debug.WriteLine("Mixture Sigma [{0},{1}] : {2}", i, l, new Matrix(_sigma[i, l]));
                            }
                        }                        
                        throw;
                    }
                }
                return _sigma;
            }
        }
    }
}
