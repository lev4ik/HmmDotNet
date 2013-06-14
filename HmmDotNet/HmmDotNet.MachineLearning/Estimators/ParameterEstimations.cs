using System.Collections.Generic;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class ParameterEstimations<TDistribution> : IParameterEstimations<TDistribution> where TDistribution : IDistribution
    {        
        private readonly IList<IObservation> _observations;
        private double[][] _coefficients;
        private readonly double[][] _alpha;
        private readonly double[][] _beta;
        private readonly IHiddenMarkovModel<TDistribution> _model;

        public ParameterEstimations(IHiddenMarkovModel<TDistribution> model, IList<IObservation> observations, double[][] alpha, double[][] beta)
        {
            _model = model;
            _observations = observations;
            _alpha = alpha;
            _beta = beta;
        }

        public int N
        {
            get
            {
                return _model.N;
            }
        }

        public int L
        {
            get
            {
                var mixture = _model.Emission[0] as Mixture<IMultivariateDistribution>; 
                if (mixture != null)
                {
                    return mixture.Components.Length;
                }
                return 0;
            }          
        }

        public double[][] Alpha
        {
            get
            {
                return _alpha;
            }
        }

        public double[][] Beta
        {
            get
            {
                return _beta;
            }
        }

        public double[][] Coefficients
        {
            get
            {
                var mixture = _model.Emission[0] as Mixture<IMultivariateDistribution>; 
                if(mixture != null)
                {
                    if (_coefficients == null)
                    {
                        _coefficients = new double[N][];
                        for (var i = 0; i < N; i++)
                        {
                            var d = _model.Emission[i] as Mixture<IMultivariateDistribution>;
                            if (d != null)
                            {
                                _coefficients[i] = d.Coefficients;   
                            }                                
                        }
                    }
                }
                return _coefficients;
            }
        }

        public IList<IObservation> Observation
        {
            get
            {
                return _observations;
            }
        }


        public IHiddenMarkovModel<TDistribution> Model
        {
            get 
            {
                return _model;
            }
        }
    }
}
