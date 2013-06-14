using System;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Algorithms.Clustering;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.Estimators;
using HmmDotNet.MachineLearning.GeneralPredictors;
using HmmDotNet.MachineLearning.GeneralPredictors.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.MachineLearning.HiddenMarkovModels.Predictors;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.MachineLearning
{
    public class HiddenMarkovModelMultivariateGaussianDistribution : HiddenMarkovModel<IMultivariateDistribution>, IMachineLearningMultivariateModel
    {
        private const double PREDICTION_LIKELIHOOD_TOLERANCE = 1d;

        #region Private Members

        private bool _initialize = false;

        #endregion Private Members

        #region Constructors

        public HiddenMarkovModelMultivariateGaussianDistribution(IModelCreationParameters<IMultivariateDistribution> parameters) : base(parameters)
        {
            _initialize = (GetModelCreationType(parameters) == HiddenMarkovModelStateCreationType.NumberOfStates ||
                           GetModelCreationType(parameters) == HiddenMarkovModelStateCreationType.NumberOfStatesAndDelta);
        }

        #endregion Constructors

        #region Predcition Methods

        public IPredictionResult Predict(PredictorType predictorType, double[][] observations, double[] weights)
        {
            var model = (HiddenMarkovModelMultivariateGaussianDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<IMultivariateDistribution> { Pi = _pi, TransitionProbabilityMatrix = _transitionProbabilityMatrix, Emissions = _emission });//new HiddenMarkovModelState<IMultivariateDistribution>(_pi, _transitionProbabilityMatrix, _emission);
            model.Normalized = Normalized;
            var request = new PredictionRequest();
            request.TrainingSet = observations;
            request.NumberOfDays = 1;
            request.Tolerance = PREDICTION_LIKELIHOOD_TOLERANCE;

            var predictor = HiddenMarkovModelPredictorFactory.GetPredictor(predictorType);
            return predictor.Predict(model, request);
        }

        public IPredictionResult Predict(PredictorType predictorType, double[][] training, double[][] test, double[] weights, int numberOfDays)
        {
            var model = (HiddenMarkovModelMultivariateGaussianDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<IMultivariateDistribution> { Pi = _pi, TransitionProbabilityMatrix = _transitionProbabilityMatrix, Emissions = _emission });//new HiddenMarkovModelState<IMultivariateDistribution>(_pi, _transitionProbabilityMatrix, _emission);
            model.Normalized = Normalized;
            var request = new PredictionRequest();
            request.TrainingSet = training;
            request.TestSet = test;
            request.NumberOfDays = numberOfDays;
            request.Tolerance = PREDICTION_LIKELIHOOD_TOLERANCE;

            var predictor = HiddenMarkovModelPredictorFactory.GetPredictor(predictorType);
            return predictor.Predict(model, request);
        }

        public IEvaluationResult EvaluatePrediction(IPredictionResult results, double[][] observations)
        {
            var model = (HiddenMarkovModelMultivariateGaussianDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<IMultivariateDistribution> { Pi = _pi, TransitionProbabilityMatrix = _transitionProbabilityMatrix, Emissions = _emission });//new HiddenMarkovModelState<IMultivariateDistribution>(_pi, _transitionProbabilityMatrix, _emission);
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

        #endregion Predictions Methods

        private void Initialize(double[][] observations)
        {
            // Create initial emmissions , TMP and Pi are already created
            var algo = new KMeans();
            algo.CreateClusters(observations, _pi.Length, KMeans.KMeansDefaultIterations, (_pi.Length > 3) ? InitialClusterSelectionMethod.Furthest : InitialClusterSelectionMethod.Random);
            _emission = new IMultivariateDistribution[_pi.Length];

            for (int i = 0; i < _pi.Length; i++)
            {
                var mean = algo.ClusterCenters[i];
                var covariance = algo.ClusterCovariances[i];

                _emission[i] = new NormalDistribution(mean, covariance);
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

            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<IMultivariateDistribution> { Pi = _pi, TransitionProbabilityMatrix = _transitionProbabilityMatrix, Emissions = _emission });//new HiddenMarkovModelState<IMultivariateDistribution>(_pi, _transitionProbabilityMatrix, _emission);
            model.Normalized = Normalized;
            var alg = new BaumWelchMultivariateDistribution(Helper.Convert(observations), model);
            var estimatedParameters = alg.Run(numberOfIterations, likelihoodTolerance);
            _pi = estimatedParameters.Pi;
            _transitionProbabilityMatrix = estimatedParameters.TransitionProbabilityMatrix;
            _emission = estimatedParameters.Emission;
            Likelihood = estimatedParameters.Likelihood;
        }

    }
}
