using System;
using System.Collections.Generic;
using System.Diagnostics;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.Estimators;
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
        private MuEstimator<IMultivariateDistribution> _muEstimator;
        private SigmaEstimator<IMultivariateDistribution> _sigmaEstimator; 

        private readonly IList<IObservation> _observations;
        private IHiddenMarkovModel<IMultivariateDistribution> _estimatedModel;
        private IHiddenMarkovModel<IMultivariateDistribution> _currentModel;

        private readonly IMultivariateDistribution[] _estimatedEmissions;

        #endregion Private Members

        #region Constructors

        [DebuggerStepThrough]
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

                // Calculate Gamma and Xi    
                // TODO : Add summing over t for xi and gamma for future calculations in differen data structure
                var parameters = new ParameterEstimations<IMultivariateDistribution>(_currentModel, _observations, forwardBackward.Alpha, forwardBackward.Beta);
                _gammaEstimator = new GammaEstimator<IMultivariateDistribution>(parameters, Normalized);
                _ksiEstimator = new KsiEstimator<IMultivariateDistribution>(parameters, Normalized);
                _muEstimator = new MuEstimator<IMultivariateDistribution>(_currentModel, _observations);
                _sigmaEstimator = new SigmaEstimator<IMultivariateDistribution>(_currentModel, _observations);
                // Estimate transition probabilities and start distribution
                EstimatePi(_gammaEstimator.Gamma);
                EstimateTransitionProbabilityMatrix(_gammaEstimator.Gamma, _ksiEstimator.Ksi, _observations.Count);
                // Estimate observation probabilities
                var muVector = _muEstimator.MuMultivariate(_gammaEstimator.Gamma);
                var sigmaVector = _sigmaEstimator.SigmaMultivariate(_gammaEstimator.Gamma, muVector);
                for (var n = 0; n < _currentModel.N; n++)
                {
                    _estimatedEmissions[n] = new NormalDistribution(muVector[n], sigmaVector[n]);
                }
                _estimatedModel = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<IMultivariateDistribution> { Pi = _estimatedPi, TransitionProbabilityMatrix = _estimatedTransitionProbabilityMatrix, Emissions = _estimatedEmissions });
                _estimatedModel.Normalized = Normalized;//new HiddenMarkovModelState<IMultivariateDistribution>(_estimatedPi, _estimatedTransitionProbabilityMatrix, _estimatedEmissions) {LogNormalized = _currentModel.LogNormalized};
                _estimatedModel.Likelihood = forwardBackward.RunForward(_observations, _estimatedModel);
                _likelihoodDelta = Math.Abs(Math.Abs(_currentModel.Likelihood) - Math.Abs(_estimatedModel.Likelihood));
                Debug.WriteLine("Iteration {3} , Current {0}, Estimate {1} Likelihood delta {2}", _currentModel.Likelihood, _estimatedModel.Likelihood, _likelihoodDelta, maxIterations);
            }
            while (!_currentModel.Equals(_estimatedModel) && maxIterations > 0 && _likelihoodDelta > likelihoodTolerance);

            return _estimatedModel;
        }

        #endregion
    }
}
