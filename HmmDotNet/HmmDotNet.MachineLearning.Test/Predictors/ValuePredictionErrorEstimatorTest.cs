using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.GeneralPredictors;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.Logic.Test.MachineLearning.Predictors
{
    [TestClass]
    public class ValuePredictionErrorEstimatorTest
    {
        private static double[][] _trainingSet;
        private static double[][] _testSet;
        private static double[][] _predictionSet;
        private static int NumberOfComponents = 2;
        private static int NumberOfStates = 4;
        private static int NumberOfIterations = 10;
        private static int LikelihoodTolerance = 20;

        [ClassInitialize]
        public static void Initializer(TestContext context)
        {
            if (_trainingSet == null)
            {
                var util = new TestDataUtils();
                _trainingSet = util.GetSvcData(util.GOOGFilePath, new DateTime(2011, 01, 01), new DateTime(2012, 01, 01));
                _testSet = util.GetSvcData(util.GOOGFilePath, new DateTime(2012, 01, 01), new DateTime(2013, 04, 05));

                var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfStates = NumberOfStates, NumberOfComponents = NumberOfComponents});//new HiddenMarkovModelMixtureDistribution(NumberOfComponents, NumberOfStates) { LogNormalized = true };
                model.Normalized = true;
                model.Train(_trainingSet, NumberOfIterations, LikelihoodTolerance);
                var result = model.Predict(PredictorType.HmmLikelihood, _trainingSet, _testSet, _testSet.Length, NumberOfIterations, LikelihoodTolerance);
                _predictionSet = result.Predicted;
            }
        }

        [TestMethod]
        public void ValuePredictionErrorEstimator_GOOGSeries_ValuePredictionErrorEstimatorCreated()
        {
            var eval = new ValuePredictionErrorEstimator(_testSet, _predictionSet);

            Assert.IsNotNull(eval);
        }

        [TestMethod]
        public void CumulativeForecastError_GOOGSeries_CumulativeForecastErrorCalculated()
        {
            var eval = new ValuePredictionErrorEstimator(_testSet, _predictionSet);
            var cfe = eval.CumulativeForecastError();

            Assert.AreEqual(52.76, cfe[0]);
            Assert.AreEqual(-19.09, cfe[1]);
            Assert.AreEqual(-169.79, cfe[2]);
            Assert.AreEqual(-270.18, cfe[3]);
        }

        [TestMethod]
        public void MeanError_GOOFSeries_MeanErrorCalculated()
        {
            var eval = new ValuePredictionErrorEstimator(_testSet, _predictionSet);
            var me = eval.MeanError();

            Assert.AreEqual(0.17, me[0]);
            Assert.AreEqual(-0.06, me[1]);
            Assert.AreEqual(-0.54, me[2]);
            Assert.AreEqual(-0.86, me[3]);
        }

        [TestMethod]
        public void MeanSquaredError_GOOGSeries_MeanSquaredErrorCalculated()
        {
            var eval = new ValuePredictionErrorEstimator(_testSet, _predictionSet);
            var mse = eval.MeanSquaredError();

            Assert.AreEqual(213.98, mse[0]);
            Assert.AreEqual(177.81, mse[1]);
            Assert.AreEqual(180.63, mse[2]);
            Assert.AreEqual(172.05, mse[3]);
        }

        [TestMethod]
        public void RootMeanSquaredError_GOOGSeries_RootMeanSquaredErrorCalculated()
        {
            var eval = new ValuePredictionErrorEstimator(_testSet, _predictionSet);
            var mse = eval.RootMeanSquaredError();

            Assert.AreEqual(14.63, mse[0]);
            Assert.AreEqual(13.33, mse[1]);
            Assert.AreEqual(13.44, mse[2]);
            Assert.AreEqual(13.12, mse[3]);
        }

        [TestMethod]
        public void MeanAbsoluteDeviation_GOOGSeries_MeanAbsoluteDeviationCalculated()
        {
            var eval = new ValuePredictionErrorEstimator(_testSet, _predictionSet);
            var mad = eval.MeanAbsoluteDeviation();

            Assert.AreEqual(10.61, mad[0]);
            Assert.AreEqual(9.39, mad[1]);
            Assert.AreEqual(9.72, mad[2]);
            Assert.AreEqual(9.59, mad[3]);
        }

        [TestMethod]
        public void MeanAbsolutePercentError_GOOGSeries_MeanAbsolutePercentErrorCalculated()
        {
            var eval = new ValuePredictionErrorEstimator(_testSet, _predictionSet);
            var mape = eval.MeanAbsolutePercentError();

            Assert.AreEqual(1.59, mape[0]);
            Assert.AreEqual(1.39, mape[1]);
            Assert.AreEqual(1.47, mape[2]);
            Assert.AreEqual(1.44, mape[3]);
        }
    }
}
