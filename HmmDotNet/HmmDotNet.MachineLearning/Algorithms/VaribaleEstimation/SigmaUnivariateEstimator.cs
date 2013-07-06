using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.Base;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class SigmaUnivariateEstimator<TDistribution> : IVariableEstimator<double[], SigmaEstimationParameters<TDistribution, double[]>>
                                                          where TDistribution : IDistribution
    {
        public double[] Estimate(SigmaEstimationParameters<TDistribution, double[]> parameters)
        {
            if (_sigmaUnivariate != null)
            {
                return _sigmaUnivariate;
            }
            _sigmaUnivariate = new double[parameters.Model.N];
            for (var n = 0; n < parameters.Model.N; n++)
            {
                var T = parameters.Observations.Count;

                var variance = 0d;
                var nK = 0d;
                for (var t = 0; t < T; t++)
                {
                    nK += parameters.Gamma[t][n];
                    var x = parameters.Observations[t].Value[0];
                    var z = x - parameters.Mean[n];
                    var m = z * z;
                    m = m * parameters.Gamma[t][n];
                    variance = variance + m;
                }
                _sigmaUnivariate[n] = variance / nK;
            }
            
            return _sigmaUnivariate;
        }
        private double[] _sigmaUnivariate;

 /*       public SigmaUnivariateEstimator(IHiddenMarkovModel<TDistribution> model, IList<IObservation> observations)
        {
            _model = model;
            _observations = observations;
        }

        private readonly IHiddenMarkovModel<TDistribution> _model;

        private readonly IList<IObservation> _observations;

        

        private double[][,] _sigmaMultivariate;
        
        public double[] SigmaUnivariate(double[][] gamma, double[] mean)
        {
            if (_sigmaUnivariate == null)
            {
                _sigmaUnivariate = new double[_model.N];
                for (var n = 0; n < _model.N; n++)
                {
                    var K = _observations[0].Dimention;
                    var T = _observations.Count;

                    var variance = 0d;
                    var nK = 0d;
                    for (var t = 0; t < T; t++)
                    {
                        nK += gamma[t][n];
                        var x = _observations[t].Value[0];
                        var z = x - mean[n];
                        var m = z * z;
                        m = m * gamma[t][n];
                        variance = variance + m;
                    }
                    _sigmaUnivariate[n] = variance / nK;
                }
            }
            return _sigmaUnivariate;
        }

        /// <summary>
        ///     Sigma[NumberOfComponents][Dimentions,Dimentions]
        /// </summary>
        /// <param name="gamma"></param>
        /// <param name="mean"></param>
        /// <returns></returns>
        public double[][,] SigmaMultivariate(double[][] gamma, double[][] mean)
        {
            if (_sigmaMultivariate == null)
            {
                try
                {
                    _sigmaMultivariate = new double[_model.N][,];
                    var K = _observations[0].Dimention;
                    var T = _observations.Count;

                    for (var n = 0; n < _model.N; n++)
                    {
                        var covariance = new double[K, K];
                        var nK = 0d;
                        for (var t = 0; t < T; t++)
                        {
                            var x = _observations[t].Value;
                            var z = x.Substruct(mean[n]);
                            var m = z.OuterProduct(z);
                            if (_model.Normalized)
                            {
                                nK += LogExtention.eExp(gamma[t][n]);
                                m = m.Product(LogExtention.eExp(gamma[t][n]));
                            }
                            else
                            {
                                nK += gamma[t][n];
                                m = m.Product(gamma[t][n]);
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
                    for (var n = 0; n < _model.N; n++)
                    {
                        Debug.WriteLine("HMM State {0} Sigma : {1}", n, new Matrix(_sigmaMultivariate[n]));
                    }                    
                    throw;
                }

            }
            return _sigmaMultivariate;
        }*/
    }
}
