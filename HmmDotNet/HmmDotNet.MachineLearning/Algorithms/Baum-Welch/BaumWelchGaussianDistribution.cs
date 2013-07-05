using System;
using System.Collections.Generic;
using System.Diagnostics;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator.EstimationParameters;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions.Univariate;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class BaumWelchGaussianDistribution : BaseBaumWelch<NormalDistribution>, IBaumWelchAlgorithm<NormalDistribution>
    {
        #region Private Members

        private GammaEstimator<NormalDistribution> _gammaEstimator;
        private KsiEstimator<NormalDistribution> _ksiEstimator;
        private MuEstimator<NormalDistribution> _muEstimator;
        private SigmaEstimator<NormalDistribution> _sigmaEstimator; 
 
        private IHiddenMarkovModel<NormalDistribution> _estimatedModel;
        private IHiddenMarkovModel<NormalDistribution> _currentModel; 
       
        private readonly IList<IObservation> _observations;

        private readonly NormalDistribution[] _estimatedEmissions;

        #endregion Private Members

        #region Constructors

        public BaumWelchGaussianDistribution(IList<IObservation> observations, IHiddenMarkovModel<NormalDistribution> model): base(model)
        {
            _currentModel = model;
            _observations = observations;

            _estimatedEmissions = new NormalDistribution[model.N];
            for (var i = 0; i < model.N; i++)
            {
                _estimatedEmissions[i] = new NormalDistribution();
            }

            _estimatedModel = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution> { Pi = _estimatedPi, TransitionProbabilityMatrix = _estimatedTransitionProbabilityMatrix, Emissions = _estimatedEmissions });//new HiddenMarkovModelState<NormalDistribution>(_estimatedPi, _estimatedTransitionProbabilityMatrix, _estimatedEmissions);
            Normalized = model.Normalized;
        }

        #endregion Constructors

        #region IBaumWelchAlgorithm Members

        public bool Normalized { get; set; }

        public IHiddenMarkovModel<NormalDistribution> Run(int maxIterations, double likelihoodTolerance)
        {
            // Initialize responce object
            var forwardBackward = new ForwardBackward(Normalized);            
            do
            {
                maxIterations--;
                if (!_estimatedModel.Likelihood.EqualsTo(0))
                {
                    _currentModel = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution> { Pi = _estimatedPi, TransitionProbabilityMatrix = _estimatedTransitionProbabilityMatrix, Emissions = _estimatedEmissions });//new HiddenMarkovModelState<NormalDistribution>(_estimatedPi, _estimatedTransitionProbabilityMatrix, _estimatedEmissions);
                    _currentModel.Normalized = Normalized;
                    _currentModel.Likelihood = _estimatedModel.Likelihood;
                }
                // Run Forward-Backward procedure
                forwardBackward.RunForward(_observations, _currentModel);
                forwardBackward.RunBackward(_observations, _currentModel);

                var @params = new AdvancedEstimationParameters<NormalDistribution>
                {
                    Alpha = forwardBackward.Alpha,
                    Beta = forwardBackward.Beta,
                    Observations = _observations,
                    Model = _currentModel,
                    Normalized = _currentModel.Normalized
                };
                _gammaEstimator = new GammaEstimator<NormalDistribution>();
                _ksiEstimator = new KsiEstimator<NormalDistribution>();
                _muEstimator = new MuEstimator<NormalDistribution>(_currentModel, _observations);
                _sigmaEstimator = new SigmaEstimator<NormalDistribution>(_currentModel, _observations);

                EstimatePi(_gammaEstimator.Estimate(@params));
                EstimateTransitionProbabilityMatrix(_gammaEstimator.Estimate(@params), _ksiEstimator.Estimate(@params), _observations.Count);
                // Estimate observation probabilities
                var muVector = _muEstimator.MuUnivariate(_gammaEstimator.Estimate(@params));
                var sigmaVector = _sigmaEstimator.SigmaUnivariate(_gammaEstimator.Estimate(@params), muVector);
                for (var j = 0; j < _currentModel.N; j++)
                {
                    _estimatedEmissions[j] = new NormalDistribution(muVector[j], sigmaVector[j]);
                }
                _estimatedModel = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution> { Pi = _estimatedPi, TransitionProbabilityMatrix = _estimatedTransitionProbabilityMatrix, Emissions = _estimatedEmissions });
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
