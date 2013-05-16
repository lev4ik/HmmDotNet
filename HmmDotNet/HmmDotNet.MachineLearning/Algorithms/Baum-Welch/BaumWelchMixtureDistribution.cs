﻿using System;
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
    public class BaumWelchMixtureDistribution : BaseBaumWelch<Mixture<IMultivariateDistribution>>, IBaumWelchAlgorithm<Mixture<IMultivariateDistribution>>
    {
        #region Private Members

        private GammaEstimator<Mixture<IMultivariateDistribution>> _gammaEstimator;
        private KsiEstimator<Mixture<IMultivariateDistribution>> _ksiEstimator;

        private readonly IList<IObservation> _observations;
        private IHiddenMarkovModelState<Mixture<IMultivariateDistribution>> _estimatedModel;
        private IHiddenMarkovModelState<Mixture<IMultivariateDistribution>> _currentModel;

        private Mixture<IMultivariateDistribution>[] _estimatedEmissions;

        #endregion Private Members

        #region Constructors

        public BaumWelchMixtureDistribution(IList<IObservation> observations, IHiddenMarkovModelState<Mixture<IMultivariateDistribution>> model): base(model)
        {
            _observations = observations;
            _currentModel = model;
            _estimatedEmissions = new Mixture<IMultivariateDistribution>[_currentModel.N];
            for (var i = 0; i < model.N; i++)
            {
                // BUG : Update emmisions from model. Don't create new ones.
                _estimatedEmissions[i] = new Mixture<IMultivariateDistribution>(model.Emission[0].Components.Length, model.Emission[0].Dimension);
            }

            _estimatedModel = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<Mixture<IMultivariateDistribution>> { Pi = _estimatedPi, TransitionProbabilityMatrix = _estimatedTransitionProbabilityMatrix, Emissions = _estimatedEmissions });//new HiddenMarkovModelState<Mixture<IMultivariateDistribution>>(_estimatedPi, _estimatedTransitionProbabilityMatrix, _estimatedEmissions);

            Normalized = _estimatedModel.Normalized = model.Normalized;
        }

        #endregion Constructors

        #region IBaumWelchAlgorithm Members

        public bool Normalized { get; set; }

        public IHiddenMarkovModelState<Mixture<IMultivariateDistribution>> Run(int maxIterations, double likelihoodTolerance)
        {
            // Initialize responce object            
            var forwardBackward = new ForwardBackward(Normalized);

            do
            {
                maxIterations--;
                if (!_estimatedModel.Likelihood.EqualsTo(0))
                {
                    _currentModel = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<Mixture<IMultivariateDistribution>> { Pi = _estimatedPi, TransitionProbabilityMatrix = _estimatedTransitionProbabilityMatrix, Emissions = _estimatedEmissions });//new HiddenMarkovModelState<Mixture<IMultivariateDistribution>>(_estimatedPi, _estimatedTransitionProbabilityMatrix, _estimatedEmissions) { LogNormalized = _estimatedModel.LogNormalized };
                    _currentModel.Normalized = Normalized;
                    _currentModel.Likelihood = _estimatedModel.Likelihood;
                }
                // Run Forward-Backward procedure
                forwardBackward.RunForward(_observations, _currentModel);
                forwardBackward.RunBackward(_observations, _currentModel);
                // Calculate Gamma and Xi 
                // TODO : Add summing over t for Ksi and gamma for future calculations in differen data structure
                var parameterEstimator = new ParameterEstimations<Mixture<IMultivariateDistribution>>(_currentModel, _observations, forwardBackward.Alpha, forwardBackward.Beta);
                _gammaEstimator = new GammaEstimator<Mixture<IMultivariateDistribution>>(parameterEstimator, Normalized);
                _ksiEstimator = new KsiEstimator<Mixture<IMultivariateDistribution>>(parameterEstimator, Normalized);
                var mixtureCoefficientsEstimator = new MixtureCoefficientsEstimator<Mixture<IMultivariateDistribution>>(parameterEstimator);
                var mixtureMuEstimator = new MixtureMuEstimator<Mixture<IMultivariateDistribution>>(parameterEstimator); // Mean
                var mixtureSigmaEstimator = new MixtureSigmaEstimator<Mixture<IMultivariateDistribution>>(parameterEstimator); // Covariance
                if (Normalized)
                {
                    mixtureCoefficientsEstimator.Denormalize();
                }
                EstimatePi(_gammaEstimator.Gamma);
                EstimateTransitionProbabilityMatrix(_gammaEstimator.Gamma, _ksiEstimator.Ksi, _observations.Count);
                for (var n = 0; n < _currentModel.N; n++)
                {
                    var mixturesComponents = _currentModel.Emission[n].Coefficients.Length;                    
                    var distributions = new IMultivariateDistribution[mixturesComponents];
                    // Calculate coefficients for state n
                    var coefficients = mixtureCoefficientsEstimator.Coefficients[n];
                    for (var l = 0; l < mixturesComponents; l++)
                    {
                        distributions[l] = new NormalDistribution(mixtureMuEstimator.Mu[n, l], mixtureSigmaEstimator.Sigma[n, l]);
                    }
                    _estimatedEmissions[n] = new Mixture<IMultivariateDistribution>(coefficients, distributions);
                }
                _estimatedModel = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<Mixture<IMultivariateDistribution>> { Pi = _estimatedPi, TransitionProbabilityMatrix = _estimatedTransitionProbabilityMatrix, Emissions = _estimatedEmissions });//new HiddenMarkovModelState<Mixture<IMultivariateDistribution>>(_estimatedPi, _estimatedTransitionProbabilityMatrix, _estimatedEmissions) { LogNormalized = _currentModel.LogNormalized };
                _estimatedModel.Normalized = Normalized;
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
