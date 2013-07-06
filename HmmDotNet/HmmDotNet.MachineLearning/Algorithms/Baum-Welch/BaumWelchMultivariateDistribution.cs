using System;
using System.Collections.Generic;
using System.Diagnostics;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimation.EstimationParameters;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class BaumWelchMultivariateDistribution : BaseBaumWelch<IMultivariateDistribution>, IBaumWelchAlgorithm<IMultivariateDistribution>
    {
        #region Private Members

        private GammaEstimator<IMultivariateDistribution> _gammaEstimator;
        private KsiEstimator<IMultivariateDistribution> _ksiEstimator;
        private MuMultivariateEstimator<IMultivariateDistribution> _muEstimator;
        private SigmaEstimator<IMultivariateDistribution> _sigmaEstimator; 

        private readonly IList<IObservation> _observations;
        private IHiddenMarkovModel<IMultivariateDistribution> _estimatedModel;
        private IHiddenMarkovModel<IMultivariateDistribution> _currentModel;

        private readonly IMultivariateDistribution[] _estimatedEmissions;

        #endregion Private Members

        #region Constructors
        
        public BaumWelchMultivariateDistribution(IList<IObservation> observations, IHiddenMarkovModel<IMultivariateDistribution> model): base(model)
        {
            _currentModel = model;
            _observations = observations;
            _estimatedEmissions = new IMultivariateDistribution[model.N];
            for (var i = 0; i < model.N; i++)
            {
                _estimatedEmissions[i] = new NormalDistribution(0);
            }

            _estimatedModel = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<IMultivariateDistribution> { Pi = _estimatedPi, TransitionProbabilityMatrix = _estimatedTransitionProbabilityMatrix, Emissions = _estimatedEmissions });//new HiddenMarkovModelState<IMultivariateDistribution>(_estimatedPi, _estimatedTransitionProbabilityMatrix, _estimatedEmissions);

            Normalized = _estimatedModel.Normalized = model.Normalized;
        }

        #endregion Constructors

        #region IBaumWelchAlgorithm Members

        public bool Normalized { get; set; }

        public IHiddenMarkovModel<IMultivariateDistribution> Run(int maxIterations, double likelihoodTolerance)
        {
            // Initialize responce object
            var forwardBackward = new ForwardBackward(Normalized);     
            do
            {
                maxIterations--;
                if (!_estimatedModel.Likelihood.EqualsTo(0))
                {
                    _currentModel = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<IMultivariateDistribution> { Pi = _estimatedPi, TransitionProbabilityMatrix = _estimatedTransitionProbabilityMatrix, Emissions = _estimatedEmissions });//new HiddenMarkovModelState<IMultivariateDistribution>(_estimatedPi, _estimatedTransitionProbabilityMatrix, _estimatedEmissions) { LogNormalized = _estimatedModel.LogNormalized };
                    _currentModel.Normalized = Normalized;
                    _currentModel.Likelihood = _estimatedModel.Likelihood;
                }
                // Run Forward-Backward procedure
                forwardBackward.RunForward(_observations, _currentModel);
                forwardBackward.RunBackward(_observations, _currentModel);

                var @params = new AdvancedEstimationParameters<IMultivariateDistribution>
                    {
                        Alpha = forwardBackward.Alpha,
                        Beta = forwardBackward.Beta,
                        Observations = _observations,
                        Model = _currentModel,
                        Normalized = _currentModel.Normalized
                    };
                _gammaEstimator = new GammaEstimator<IMultivariateDistribution>();
                _ksiEstimator = new KsiEstimator<IMultivariateDistribution>();
                _muEstimator = new MuMultivariateEstimator<IMultivariateDistribution>();
                _sigmaEstimator = new SigmaEstimator<IMultivariateDistribution>(_currentModel, _observations);

                EstimatePi(_gammaEstimator.Estimate(@params));
                EstimateTransitionProbabilityMatrix(_gammaEstimator.Estimate(@params), _ksiEstimator.Estimate(@params), _observations.Count);
                // Estimate observation probabilities
                var muParams = new MuEstimationParameters<IMultivariateDistribution>
                    {
                        Gamma = _gammaEstimator.Estimate(@params),
                        Model = _currentModel,
                        Normalized = _currentModel.Normalized,
                        Observations = _observations
                    };

                var muVector = _muEstimator.Estimate(muParams);
                var sigmaVector = _sigmaEstimator.SigmaMultivariate(_gammaEstimator.Estimate(@params), muVector);
                for (var n = 0; n < _currentModel.N; n++)
                {
                    _estimatedEmissions[n] = new NormalDistribution(muVector[n], sigmaVector[n]);
                }
                _estimatedModel = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<IMultivariateDistribution> { Pi = _estimatedPi, TransitionProbabilityMatrix = _estimatedTransitionProbabilityMatrix, Emissions = _estimatedEmissions });
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
