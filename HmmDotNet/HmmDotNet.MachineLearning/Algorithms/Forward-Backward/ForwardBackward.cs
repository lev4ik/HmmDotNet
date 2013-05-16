using System.Collections.Generic;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class ForwardBackward
    {
        #region Constructors

        public ForwardBackward(bool logNormalized)
        {
            Normalized = logNormalized;
        }

        #endregion Constructors

        #region Private

        private double[][] _alpha;
        private double[][] _beta;

        #endregion Private

        #region Public Variables

        public bool Normalized { get; set; }

        public double[][] Alpha
        {
            get { return _alpha; }
        }
        
        public double[][] Beta
        {
            get { return _beta; }
        }

        #endregion Public Variables

        public double RunForward<TEmmisionType>(IList<IObservation> observations, 
                                                IHiddenMarkovModelState<TEmmisionType> model) where TEmmisionType : IDistribution
        {
            var alphaEstimator = new AlphaEstimator<TEmmisionType>(model, 
                                                                   observations, 
                                                                   Normalized);
            _alpha = alphaEstimator.Alpha;
            var T = observations.Count;
            var result = (Normalized) ? double.NaN : 0d;

            // Calculate results
            for (var i = 0; i < model.N; i++)
            {
                if (Normalized)
                {
                    result = LogExtention.eLnSum(result, _alpha[T - 1][i]);
                }
                else
                {
                    result += _alpha[T - 1][i];
                }
            }
            
            return result;
        }

        public double RunBackward<TEmmisionType>(IList<IObservation> observations, IHiddenMarkovModelState<TEmmisionType> model) where TEmmisionType : IDistribution
        {
            var betaEstimator = new BetaEstimator<TEmmisionType>(model, observations, Normalized);
            _beta = betaEstimator.Beta;
            var result = (Normalized) ? double.NaN : 0d;

            for (var j = 0; j < model.N; j++)
            {
                if (Normalized)
                {
                    result = LogExtention.eLnSum(result,
                                                  LogExtention.eLnProduct(LogExtention.eLn(model.Pi[j]), _beta[1][j]));
                }
                else
                {
                    result += model.Pi[j] * _beta[1][j];
                }
            }

            return result;
        }
    }
}
