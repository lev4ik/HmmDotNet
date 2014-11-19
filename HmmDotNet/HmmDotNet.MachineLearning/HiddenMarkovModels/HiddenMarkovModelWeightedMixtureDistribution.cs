using HmmDotNet.MachineLearning.Algorithms.Baum_Welch;
using HmmDotNet.MachineLearning.Algorithms.Clustering;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.Estimators;
using HmmDotNet.MachineLearning.GeneralPredictors;
using HmmDotNet.MachineLearning.GeneralPredictors.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels.Predictors;
using HmmDotNet.Mathematic;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace HmmDotNet.MachineLearning.HiddenMarkovModels
{
    public class HiddenMarkovModelWeightedMixtureDistribution : HiddenMarkovModel<Mixture<IMultivariateDistribution>>, IMachineLearningMultivariateModel
    {
        private const double PREDICTION_LIKELIHOOD_TOLERANCE = 1d;

        #region Private Members

        private bool _initialize = false;

        #endregion Private Members

        #region Constructors

        public HiddenMarkovModelWeightedMixtureDistribution(IModelCreationParameters<Mixture<IMultivariateDistribution>> parameters)
            : base(parameters)
        {
            _initialize = (GetModelCreationType(parameters) == HiddenMarkovModelStateCreationType.NumberOfStatesAndNumberOfComponents ||
                           GetModelCreationType(parameters) == HiddenMarkovModelStateCreationType.NumberOfStatesAndDeltaAndNumberOfComponents);
        }

        #endregion Constructor

        private void Initialize(double[][] observations)
        {
            var algo = new KMeans();
            var k = _pi.Length * _numberOfComponents;
            var dimentions = observations[0].Length;
            algo.CreateClusters(observations, k, KMeans.KMeansDefaultIterations, (k > 3) ? InitialClusterSelectionMethod.Furthest : InitialClusterSelectionMethod.Random);
            _emission = new Mixture<IMultivariateDistribution>[_pi.Length];

            for (int i = 0; i < _pi.Length; i++)
            {
                _emission[i] = new Mixture<IMultivariateDistribution>(_numberOfComponents, dimentions);
                for (int j = 0; j < _numberOfComponents; j++)
                {
                    var mean = algo.ClusterCenters[j + _numberOfComponents * i];
                    var covariance = algo.ClusterCovariances[j + _numberOfComponents * i];

                    _emission[i].Components[j] = new NormalDistribution(mean, covariance);
                    Debug.WriteLine("[i,j]=[{0},{1}] Mean Vector {2}",i,j, new Vector(mean));
                }
            }
        }

        public void Train(double[][] observations, int numberOfIterations, double likelihoodTolerance)
        {
            if (_initialize)
            {
                Initialize(observations);
            }
            if (_pi == null || _transitionProbabilityMatrix == null || _emission == null)
            {
                throw new ApplicationException("Initialize the model with initial valuesss");
            }
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<Mixture<IMultivariateDistribution>> { Pi = _pi, TransitionProbabilityMatrix = _transitionProbabilityMatrix, Emissions = _emission });//new HiddenMarkovModelState<Mixture<IMultivariateDistribution>>(_pi, _transitionProbabilityMatrix, _emission);
            model.Normalized = Normalized;
            var alg = new BaumWelchWeightedMixtureDistribution(Helper.Convert(observations),
                                                               TimeSensitiveWeightCalculator.Calculate(7, observations.Length), 
                                                               model);
            var estimatedParameters = alg.Run(numberOfIterations, likelihoodTolerance);
            _pi = estimatedParameters.Pi;
            _transitionProbabilityMatrix = estimatedParameters.TransitionProbabilityMatrix;
            _emission = estimatedParameters.Emission;
            Likelihood = estimatedParameters.Likelihood;
        }

        #region Prediction Methods

        public IPredictionResult Predict(PredictorType predictorType, double[][] observations, int numberOfIterations, double likelihoodTolerance)
        {
            var model = (HiddenMarkovModelWeightedMixtureDistribution)HiddenMarkovModelFactory.GetModel<HiddenMarkovModelWeightedMixtureDistribution, Mixture<IMultivariateDistribution>>(new ModelCreationParameters<Mixture<IMultivariateDistribution>> { Pi = _pi, TransitionProbabilityMatrix = _transitionProbabilityMatrix, Emissions = _emission });//HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<Mixture<IMultivariateDistribution>> { Pi = _pi, TransitionProbabilityMatrix = _transitionProbabilityMatrix, Emissions = _emission });
            model.Normalized = Normalized;
            var request = new PredictionRequest();
            request.TrainingSet = observations;
            request.NumberOfDays = 1;
            request.Tolerance = PREDICTION_LIKELIHOOD_TOLERANCE;
            request.TrainingLikelihoodTolerance = likelihoodTolerance;
            request.NumberOfTrainingIterations = numberOfIterations;

            var predictor = HiddenMarkovModelPredictorFactory.GetPredictor(predictorType);
            return predictor.Predict(model, request);
        }

        public IPredictionResult Predict(PredictorType predictorType, double[][] training, double[][] test, int numberOfDays, int numberOfIterations, double likelihoodTolerance)
        {
            var model = (HiddenMarkovModelWeightedMixtureDistribution)HiddenMarkovModelFactory.GetModel<HiddenMarkovModelWeightedMixtureDistribution, Mixture<IMultivariateDistribution>>(new ModelCreationParameters<Mixture<IMultivariateDistribution>> { Pi = _pi, TransitionProbabilityMatrix = _transitionProbabilityMatrix, Emissions = _emission });//new HiddenMarkovModelState<IMultivariateDistribution>(_pi, _transitionProbabilityMatrix, _emission);
            model.Normalized = Normalized;
            var request = new PredictionRequest();
            request.TrainingSet = training;
            request.TestSet = test;
            request.NumberOfDays = numberOfDays;
            request.Tolerance = PREDICTION_LIKELIHOOD_TOLERANCE;
            request.TrainingLikelihoodTolerance = likelihoodTolerance;
            request.NumberOfTrainingIterations = numberOfIterations;

            var predictor = HiddenMarkovModelPredictorFactory.GetPredictor(predictorType);
            return predictor.Predict(model, request);
        }

        public IEvaluationResult EvaluatePrediction(IPredictionResult results, double[][] observations)
        {
            var model = (HiddenMarkovModelWeightedMixtureDistribution)HiddenMarkovModelFactory.GetModel<HiddenMarkovModelWeightedMixtureDistribution, Mixture<IMultivariateDistribution>>(new ModelCreationParameters<Mixture<IMultivariateDistribution>> { Pi = _pi, TransitionProbabilityMatrix = _transitionProbabilityMatrix, Emissions = _emission });//new HiddenMarkovModelState<IMultivariateDistribution>(_pi, _transitionProbabilityMatrix, _emission);
            model.Normalized = Normalized;
            var request = new EvaluationRequest();
            request.PredictionParameters = new PredictionRequest();
            request.PredictionToEvaluate = new PredictionResult();
            request.EstimatorType = ErrorEstimatorType.Value;
            request.PredictionParameters.Tolerance = PREDICTION_LIKELIHOOD_TOLERANCE;
            request.PredictionParameters.TestSet = observations;
            request.PredictionToEvaluate.Predicted = results.Predicted;

            var predictor = new LikelihoodBasedPredictor();
            return predictor.Evaluate(request);
        }

        #endregion Prediction Methods

    }
}
