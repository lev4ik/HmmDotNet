using System;
using System.Collections.Generic;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.GeneralPredictors;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.MachineLearning.HiddenMarkovModels.Predictors;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HmmDotNet.MachineLearning.Test.Predictors
{
    [TestClass]
    public class SearchBasedPredictorTest
    {
        private const int _NumberOfComponents = 2;
        private const int _NumberOfIterations = 10;
        private const int _NumberOfStates = 2;
        private const int _LikelihoodTolerance = 20;

        [TestMethod]
        public void Predict_PredictionRequestWithoutParameters_Null()
        {
            var predictor = new SearchBasedPredictor(null);

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStates });
            var request = new SearchBasedPredictionRequest();

            var value = predictor.Predict(model, request);

            Assert.IsNull(value);
        }

        [TestMethod]
        public void Predict_PredictionRequestWihtoutNumberOfSamplePoints_Null()
        {
            var predictor = new SearchBasedPredictor(null);

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStates });
            var request = new SearchBasedPredictionRequest();
            request.AlgorithmSpecificParameters = new Dictionary<string, string>();

            var value = predictor.Predict(model, request);

            Assert.IsNull(value);            
        }

        [TestMethod]
        public void Predict_PredictionRequestWihtoutNumberOfWinningPoints_Null()
        {
            var predictor = new SearchBasedPredictor(null);

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStates });
            var request = new SearchBasedPredictionRequest();
            request.AlgorithmSpecificParameters = new Dictionary<string, string>
                {
                    {"NumberOfSamplePoints", "0"}
                };

            var value = predictor.Predict(model, request);

            Assert.IsNull(value);
        }

        [TestMethod]
        public void Predict_PredictionRequestWihtoutNumberOfPredictionIterations_Null()
        {
            var predictor = new SearchBasedPredictor(null);

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStates });
            var request = new SearchBasedPredictionRequest();
            request.AlgorithmSpecificParameters = new Dictionary<string, string>
                {
                    {"NumberOfSamplePoints", "0"},
                    {"NumberOfWinningPoints", "0"}
                };

            var value = predictor.Predict(model, request);

            Assert.IsNull(value);
        }

        [TestMethod]
        public void Predict_PredictionRequestWihtoutEpsilon_Null()
        {
            var predictor = new SearchBasedPredictor(null);

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStates });
            var request = new SearchBasedPredictionRequest();
            request.AlgorithmSpecificParameters = new Dictionary<string, string>
                {
                    {"NumberOfSamplePoints", "0"},
                    {"NumberOfWinningPoints", "0"},
                    {"NumberOfPredictionIterations", "0"}
                };

            var value = predictor.Predict(model, request);

            Assert.IsNull(value);
        }

        [TestMethod]
        public void Predict_ModelAndPredictionRequest_PredictedValue()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = _NumberOfComponents, NumberOfStates = _NumberOfStates });
            model.Normalized = true;
            model.Train(series, _NumberOfIterations, _LikelihoodTolerance);
            var request = new SearchBasedPredictionRequest();
            request.TrainingSet = series;
            request.NumberOfDays = 1;


            var predictor = new SearchBasedPredictor(null);

            var value = predictor.Predict(model, request);

            Assert.AreEqual(0, value);
        }
    }
}
