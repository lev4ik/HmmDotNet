using System;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.Estimators;
using HmmDotNet.MachineLearning.GeneralPredictors;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.MachineLearning.HiddenMarkovModels.Predictors;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Univariate;

namespace HmmDotNet.MachineLearning
{
    public class HiddenMarkovModel : HiddenMarkovModel<DiscreteDistribution>, IMachineLearningUnivariateModel
    {
        #region Constructors

        public HiddenMarkovModel(IModelCreationParameters<DiscreteDistribution> parameters)
            : base(parameters)
        {

        }

        #endregion Constructors

        public IPredictionResult Predict(double[][] observations, double[] weights)
        {
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<DiscreteDistribution> { Pi = _pi, TransitionProbabilityMatrix = _transitionProbabilityMatrix, Emissions = _emission });//new HiddenMarkovModelState<IMultivariateDistribution>(_pi, _transitionProbabilityMatrix, _emission);

            var request = new PredictionRequest();
            request.TrainingSet = observations;
            request.NumberOfDays = 1;
            request.Tolerance = 0.01;

            var predictor = new LikelihoodBasedPredictor();
            return predictor.Predict((IHiddenMarkovModel<IDistribution>)model, request);
        }

        public IPredictionResult Predict(double[][] training, double[][] test, double[] weights, int numberOfDays)
        {
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<DiscreteDistribution> { Pi = _pi, TransitionProbabilityMatrix = _transitionProbabilityMatrix, Emissions = _emission });//new HiddenMarkovModelState<IMultivariateDistribution>(_pi, _transitionProbabilityMatrix, _emission);

            var request = new PredictionRequest();
            request.TrainingSet = training;
            request.TestSet = test;
            request.NumberOfDays = 1;
            request.Tolerance = 0.01;

            var predictor = new LikelihoodBasedPredictor();
            return predictor.Predict((IHiddenMarkovModel<IDistribution>)model, request);
        }

        public void Train(double[] observations)
        {
            if (_pi == null || _transitionProbabilityMatrix == null || _emission == null)
            {
                throw new ApplicationException("Initialize the model with initial valuesss");
            }
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<DiscreteDistribution> { Pi = _pi, TransitionProbabilityMatrix = _transitionProbabilityMatrix, Emissions = _emission });//new HiddenMarkovModelState<DiscreteDistribution>(_pi, _transitionProbabilityMatrix, _emission);
            model.Normalized = Normalized;
            var alg = new BaumWelch(Helper.Convert(observations), model, Helper.Convert(observations));
            var estimatedParameters = alg.Run(100, 20);
            _pi = estimatedParameters.Pi;
            _transitionProbabilityMatrix = estimatedParameters.TransitionProbabilityMatrix;
            _emission = estimatedParameters.Emission;
            Likelihood = estimatedParameters.Likelihood;
        }
    }
}
