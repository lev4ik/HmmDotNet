using System;
using System.Collections.Generic;
using System.Diagnostics;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions.Univariate;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class BaumWelch : BaseBaumWelch<DiscreteDistribution>, IBaumWelchAlgorithm<DiscreteDistribution>
    {
        #region Private Members

        private GammaEstimator<DiscreteDistribution> _gammaEstimator;
        private KsiEstimator<DiscreteDistribution> _ksiEstimator;

        private IHiddenMarkovModel<DiscreteDistribution> _estimatedModel;
        private IHiddenMarkovModel<DiscreteDistribution> _currentModel; 
       
        private readonly IList<IObservation> _observations;
        
        private readonly double[] _discreteSymbols;
        private readonly double[] _discreteObservations;

        private readonly DiscreteDistribution[] _estimatedEmissions;

        #endregion Private Members

        #region Constructors

        public BaumWelch(IList<IObservation> observations, IHiddenMarkovModel<DiscreteDistribution> model, IList<IObservation> symbols) : base(model)
        {
            _currentModel = model;
            
            _observations = observations;

            _discreteSymbols = new double[_currentModel.M];
            _discreteObservations = new double[_observations.Count];

            for (var i = 0; i < _currentModel.M; i++)
            {
                _discreteSymbols[i] = symbols[i].Value[0];
            }
            for (var i = 0; i < _observations.Count; i++)
            {
                _discreteObservations[i] = observations[i].Value[0];
            }

            _estimatedEmissions = new DiscreteDistribution[_currentModel.N];
            for (var i = 0; i < _currentModel.N; i++)
            {
                _estimatedEmissions[i] = new DiscreteDistribution(_discreteSymbols, _discreteObservations);
            }
            _estimatedModel = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<DiscreteDistribution> { Pi = _estimatedPi, TransitionProbabilityMatrix = _estimatedTransitionProbabilityMatrix, Emissions = _estimatedEmissions });//new HiddenMarkovModelState<DiscreteDistribution>(_estimatedPi, _estimatedTransitionProbabilityMatrix, _estimatedEmissions);
            Normalized = model.Normalized;
        }

        #endregion Constructors
        
        #region IBaumWelchAlgorithm Members

        public bool Normalized { get; set; }

        public IHiddenMarkovModel<DiscreteDistribution> Run(int maxIterations, double likelihoodTolerance)
        {
            // Initialize responce object
            var forwardBackward = new ForwardBackward(Normalized);
            do
            {
                maxIterations--;
                if (!_estimatedModel.Likelihood.EqualsTo(0))
                {
                    _currentModel = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<DiscreteDistribution> { Pi = _estimatedPi, TransitionProbabilityMatrix = _estimatedTransitionProbabilityMatrix, Emissions = _estimatedEmissions });//new HiddenMarkovModelState<DiscreteDistribution>(_estimatedPi, _estimatedTransitionProbabilityMatrix, _estimatedEmissions);
                    _currentModel.Normalized = Normalized;
                    _currentModel.Likelihood = _estimatedModel.Likelihood;
                }
                // Run Forward-Backward procedure
                forwardBackward.RunForward(_observations, _currentModel);
                forwardBackward.RunBackward(_observations, _currentModel);

                var parameters = new ParameterEstimations<DiscreteDistribution>(_currentModel, _observations, forwardBackward.Alpha, forwardBackward.Beta);
                _gammaEstimator = new GammaEstimator<DiscreteDistribution>(parameters, Normalized);
                _ksiEstimator = new KsiEstimator<DiscreteDistribution>(parameters, Normalized);
                
                // Estimate transition probabilities and start distribution
                EstimatePi(_gammaEstimator.Gamma);
                EstimateTransitionProbabilityMatrix(_gammaEstimator.Gamma, _ksiEstimator.Ksi, _observations.Count);
                // Estimate Emmisions
                for (var j = 0; j < _currentModel.N; j++)
                {
                    _estimatedEmissions[j] = (DiscreteDistribution)_estimatedEmissions[j].Evaluate(_discreteObservations, _discreteSymbols, _gammaEstimator.Gamma.GetColumn(j), Normalized);
                }

                _estimatedModel = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<DiscreteDistribution> { Pi = _estimatedPi, TransitionProbabilityMatrix = _estimatedTransitionProbabilityMatrix, Emissions = _estimatedEmissions });
                _estimatedModel.Normalized = Normalized;
                _estimatedModel.Likelihood = forwardBackward.RunForward(_observations, _estimatedModel);
                _likelihoodDelta = Math.Abs(Math.Abs(_currentModel.Likelihood) - Math.Abs(_estimatedModel.Likelihood));
                Debug.WriteLine("Iteration {3} , Current {0}, Estimate {1} Likelihood delta {2}", _currentModel.Likelihood, _estimatedModel.Likelihood, _likelihoodDelta, maxIterations);
            }
            while (_currentModel != _estimatedModel && maxIterations > 0 && _likelihoodDelta > likelihoodTolerance);

            return _estimatedModel;
        }
       
        #endregion
    }
}
