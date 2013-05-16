using System;
using System.Diagnostics;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    /// <summary>
    ///     Calculates Gamma value for all t from 1 to T and for all i from 1 to N
    /// </summary>
    public class GammaEstimator<TDistribution> : BaseEstimator where TDistribution : IDistribution
    {
        public GammaEstimator(IParameterEstimations<TDistribution> parameters, bool logNormalized)
        {
            LogNormalized = logNormalized;
            Parameters = parameters;
            CalculateDenominator();
            CalculateGamma();
        }

        private readonly IParameterEstimations<TDistribution> Parameters;

        private double[] _denominator;

        private double[][] _gamma;

        protected double[] Denominator 
        { 
            get
            {
                return _denominator;
            }
        }

        public double[][] Gamma
        {
            get
            {
                return _gamma;
            }
        }

        #region Private Methods

        private void CalculateDenominator()
        {
            if (_denominator == null)
            {
                _denominator = new double[Parameters.Observation.Count];
                for (var t = 0; t < Parameters.Observation.Count; t++)
                {
                    _denominator[t] = (LogNormalized) ? double.NaN : 0d;
                    for (var i = 0; i < Parameters.Model.N; i++)
                    {
                        if (LogNormalized)
                        {
                            // TODO : Check if Alpha and Beta is already passed Ln function
                            _denominator[t] = LogExtention.eLnSum(_denominator[t], LogExtention.eLnProduct(Parameters.Alpha[t][i], Parameters.Beta[t][i]));
                        }
                        else
                        {
                            _denominator[t] += Parameters.Alpha[t][i] * Parameters.Beta[t][i];
                        }
                    }
                }
            }            
        }

        private void CalculateGamma()
        {
            if (_gamma == null)
            {
                try
                {
                    _gamma = new double[Parameters.Observation.Count][];
                    for (var t = 0; t < Parameters.Observation.Count; t++)
                    {
                        _gamma[t] = new double[Parameters.Model.N];
                        for (var i = 0; i < Parameters.Model.N; i++)
                        {
                            if (LogNormalized)
                            {
                                _gamma[t][i] = LogExtention.eLnProduct(LogExtention.eLnProduct(Parameters.Alpha[t][i], Parameters.Beta[t][i]), -Denominator[t]);
                            }
                            else
                            {
                                _gamma[t][i] = (Parameters.Alpha[t][i] * Parameters.Beta[t][i]) / Denominator[t];
                            }
                        }
                    }
                }
                catch (Exception)
                {
                    for (var t = 0; t < Parameters.Observation.Count; t++)
                    {
                        for (var i = 0; i < Parameters.Model.N; i++)
                        {
                            Debug.WriteLine("Gamma [{0}][{1}] : alpha : {2} , beta : {3} , denominator : {4} : gamma {5} ", t, i, Parameters.Alpha[t][i], Parameters.Beta[t][i], Denominator[t], _gamma[t][i]);
                        }
                    }                    
                    throw;
                }
            }
        }

        #endregion Private Methods
    }
}
