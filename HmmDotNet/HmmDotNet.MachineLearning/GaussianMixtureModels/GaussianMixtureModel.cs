using HmmDotNet.MachineLearning.Algorithms.Clustering;
using HmmDotNet.MachineLearning.GeneralPredictors;
using HmmDotNet.MachineLearning.GeneralPredictors.Base;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;
using HmmDotNet.MachineLearning.Base;
using LikelihoodBasedPredictor = HmmDotNet.MachineLearning.GaussianMixtureModels.Predictors.LikelihoodBasedPredictor;

namespace HmmDotNet.MachineLearning.GaussianMixtureModels
{
    /// <summary>
    ///     Gaussian Mixture Model
    /// </summary>
    public class GaussianMixtureModel : IGaussianMixtureModelState, IMachineLearningMultivariateModel
    {
        private const double PREDICTION_LIKELIHOOD_TOLERANCE = 0.01;

        #region Private Variables

        private Mixture<IMultivariateDistribution> _mixture;
        private double _likelihood = double.NaN;
        private bool _initialize = false;

        #endregion Private Variables

        #region Constructors

        /// <summary>
        ///     Initializes a new instance of the GaussianMixtureModel class
        /// </summary>
        /// <param name="coeficient">Components selecting distribution</param>
        /// <param name="components">Distribution components</param>
        public GaussianMixtureModel(double[] coeficient, IMultivariateDistribution[] components)
        {
            _mixture = new Mixture<IMultivariateDistribution>(coeficient, components);
        }

        public GaussianMixtureModel(int components, int dimentions)
        {
            _mixture = new Mixture<IMultivariateDistribution>(components, dimentions);
            _initialize = true;
        }

        #endregion Constructors

        #region Initialization

        private void Initialize(double[][] observations)
        {
            // Create new Emty components            
            var covariance = new double[_mixture.Dimension, _mixture.Dimension];
            var mean = new double[_mixture.Dimension];
            var algo = new KMeans();
            algo.CreateClusters(observations, _mixture.Components.Length, KMeans.KMeansDefaultIterations, (_mixture.Components.Length > 3) ? InitialClusterSelectionMethod.Furthest : InitialClusterSelectionMethod.Random);
            
            for (int i = 0; i < _mixture.Components.Length; i++)
            {
                mean = algo.ClusterCenters[i];
                covariance = algo.ClusterCovariances[i];

                _mixture.Components[i] = new NormalDistribution(mean, covariance);
            }
        }

        #endregion Initialization

        #region Properties

        public double Likelihood
        {
            get { return _likelihood; }
        }

        public Mixture<IMultivariateDistribution> Mixture
        {
            get { return _mixture; }
        }

        #endregion Properties

        #region Methods

        /// <summary>
        ///     Predicts next value for the time series. 
        ///     The prediction is based on the distance of last day observation from the observation 
        ///     with closest likelihood.
        /// </summary>
        /// <param name="observations">Observation matrix. As a static database of observations</param>
        /// <param name="weights">Observation weights</param>
        /// <returns>Predicted vector</returns>
        public IPredictionResult Predict(double[][] observations, double[] weights)
        {
            var model = new GaussianMixtureModel(_mixture.Coefficients, _mixture.Components);
            var request = new PredictionRequest();
            request.TrainingSet = observations;
            request.NumberOfDays = 1;
            request.Tolerance = PREDICTION_LIKELIHOOD_TOLERANCE;

            var predictor = new LikelihoodBasedPredictor();
            return predictor.Predict(model, request);
        }

        public IPredictionResult Predict(double[][] training, double[][] test, double[] weights, int numberOfDays)
        {
            var model = new GaussianMixtureModel(_mixture.Coefficients, _mixture.Components);
            var request = new PredictionRequest();
            request.TrainingSet = training;
            request.TestSet = test;
            request.NumberOfDays = numberOfDays;
            request.Tolerance = PREDICTION_LIKELIHOOD_TOLERANCE;

            var predictor = new LikelihoodBasedPredictor();
            return predictor.Predict(model, request);
        }

        public IEvaluationResult EvaluatePrediction(IPredictionResult predicted, double[][] observed)
        {
            var request = new EvaluationRequest();
            request.PredictionParameters = new PredictionRequest();
            request.PredictionToEvaluate = new PredictionResult();
            request.EstimatorType = ErrorEstimatorType.Value;
            request.PredictionParameters.Tolerance = PREDICTION_LIKELIHOOD_TOLERANCE;
            request.PredictionParameters.TestSet = observed;
            request.PredictionToEvaluate.Predicted = predicted.Predicted;

            var predictor = new LikelihoodBasedPredictor();
            return predictor.Evaluate(request);
        }
        /// <summary>
        ///     Trains Gaussian Mixture Model
        /// </summary>
        /// <param name="observations">Observation matrix</param>
        /// <param name="numberOfIterations">Number Of Iterations</param>
        public void Train(double[][] observations, int numberOfIterations, double likelihoodTolerance)
        {
            if (_initialize)
            {
                Initialize(observations);
            }
            _mixture = (Mixture<IMultivariateDistribution>)_mixture.Evaluate(observations, out _likelihood);
        }

        #endregion Methods
    }
}
