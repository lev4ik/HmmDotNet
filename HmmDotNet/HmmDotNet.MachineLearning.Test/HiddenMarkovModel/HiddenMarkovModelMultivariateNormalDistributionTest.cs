using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning;
using HmmDotNet.MachineLearning.Algorithms.Clustering;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.Logic.Test.MachineLearning.HiddenMarkovModel
{
    [TestClass]
    public class HiddenMarkovModelMultivariateNormalDistributionTest
    {
        private const int NumberOfStates = 2;
        private const int NumberOfStatesRightLeft = 4;
        private const int NumberOfIterations = 10;
        private const int LikelihoodTolerance = 20;
        private const int Delta = 3;

        private NormalDistribution[] CreateEmissions(double[][] observations, int numberOfEmissions)
        {
            var emissions = new NormalDistribution[numberOfEmissions];
            // Create initial emmissions , TMP and Pi are already created
            var algo = new KMeans();
            algo.CreateClusters(observations, numberOfEmissions, KMeans.KMeansDefaultIterations, (numberOfEmissions > 3) ? InitialClusterSelectionMethod.Furthest : InitialClusterSelectionMethod.Random);

            for (int i = 0; i < numberOfEmissions; i++)
            {
                var mean = algo.ClusterCenters[i];
                var covariance = algo.ClusterCovariances[i];

                emissions[i] = new NormalDistribution(mean, covariance);
            }

            return emissions;
        }

        [TestMethod]
        public void HiddenMarkovModelMultivariateGaussianDistribution_NumberOfStates_ErgodicModelCreated()
        {
            var model = HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<IMultivariateDistribution>() { NumberOfStates = NumberOfStates });//new HiddenMarkovModelMultivariateGaussianDistribution(NumberOfStates);

            Assert.AreEqual(ModelType.Ergodic, model.Type);
            Assert.AreEqual(NumberOfStates, model.N);
            Assert.IsFalse(model.Normalized);
            Assert.IsNull(model.Emission);
            for (int n = 0; n < NumberOfStates; n++)
            {
                Assert.AreEqual(0.5, model.Pi[n]);
                for (int i = 0; i < NumberOfStates; i++)
                {
                    Assert.AreEqual(0.5, model.TransitionProbabilityMatrix[n][i]);
                }
            }
        }

        [TestMethod]
        public void HiddenMarkovModelMultivariateGaussianDistribution_NumberOfStatesAndDelta_RightLeftModelCreated()
        {
            var model = HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<IMultivariateDistribution>() { NumberOfStates = NumberOfStatesRightLeft, Delta = Delta });//new HiddenMarkovModelMultivariateGaussianDistribution(NumberOfStatesRightLeft, Delta);

            Assert.AreEqual(ModelType.LeftRight, model.Type);
            Assert.AreEqual(NumberOfStatesRightLeft, model.N);
            Assert.IsFalse(model.Normalized);
            Assert.IsNull(model.Emission);
            Assert.AreEqual(1d, model.Pi[0]);
            Assert.AreEqual(0.25, model.TransitionProbabilityMatrix[0][0]);
            Assert.AreEqual(0.25, model.TransitionProbabilityMatrix[0][1]);
            Assert.AreEqual(0.25, model.TransitionProbabilityMatrix[0][2]); 
            Assert.AreEqual(0.25, model.TransitionProbabilityMatrix[0][3]);
            Assert.AreEqual(0d, model.TransitionProbabilityMatrix[1][0]);
            Assert.AreEqual(1d/3d, model.TransitionProbabilityMatrix[1][1]);
            Assert.AreEqual(1d/3d, model.TransitionProbabilityMatrix[1][2]);
            Assert.AreEqual(1d/3d, model.TransitionProbabilityMatrix[1][3]);
            Assert.AreEqual(0, model.TransitionProbabilityMatrix[2][0]);
            Assert.AreEqual(0, model.TransitionProbabilityMatrix[2][1]);
            Assert.AreEqual(0.5, model.TransitionProbabilityMatrix[2][2]);
            Assert.AreEqual(0.5, model.TransitionProbabilityMatrix[2][3]);
            Assert.AreEqual(0, model.TransitionProbabilityMatrix[3][0]);
            Assert.AreEqual(0, model.TransitionProbabilityMatrix[3][1]);
            Assert.AreEqual(0, model.TransitionProbabilityMatrix[3][2]);
            Assert.AreEqual(1d, model.TransitionProbabilityMatrix[3][3]);
        }

        [TestMethod]
        public void HiddenMarkovModelMultivariateGaussianDistribution_ModelState_ModelCreated()
        {
            var pi = new double[NumberOfStates] { 0.5, 0.5 };
            var tpm = new double[NumberOfStates][] { new []{ 0.5, 0.5 }, new []{ 0.5, 0.5 } };
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var emissions = CreateEmissions(series, NumberOfStates);

            var model = HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<IMultivariateDistribution>() { Pi = pi, TransitionProbabilityMatrix = tpm, Emissions = emissions });//new HiddenMarkovModelMultivariateGaussianDistribution(pi, tpm, emissions);

            Assert.AreEqual(ModelType.Ergodic, model.Type);
            Assert.AreEqual(NumberOfStates, model.N);
            Assert.IsFalse(model.Normalized);
            for (int n = 0; n < NumberOfStates; n++)
            {
                Assert.AreEqual(0.5, model.Pi[n]);
                Assert.IsNotNull(model.Emission[n]);
                Assert.IsInstanceOfType(model.Emission[n], typeof(NormalDistribution));
                for (int i = 0; i < NumberOfStates; i++)
                {
                    Assert.AreEqual(0.5, model.TransitionProbabilityMatrix[n][i]);
                }
            }
        }

        [TestMethod]
        public void Train_FTSESeriesAndErgodicModel_TrainedModel()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));

            var model = (HiddenMarkovModelMultivariateGaussianDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<IMultivariateDistribution>() { NumberOfStates = NumberOfStates });//new HiddenMarkovModelMultivariateGaussianDistribution(NumberOfStates) {LogNormalized = true};
            model.Normalized = true;
            model.Train(series, NumberOfIterations, LikelihoodTolerance);

            Assert.AreEqual(ModelType.Ergodic, model.Type);
            Assert.AreEqual(NumberOfStates, model.Emission.Length);
            Assert.IsTrue(model.Normalized);
            for (int n = 0; n < NumberOfStates; n++)
            {
                Assert.IsInstanceOfType(model.Emission[n], typeof(NormalDistribution));
            }
        }

        [TestMethod]
        public void Train_FTSESeriesAndLeftRightModel_TrainedModel()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));

            var model = (HiddenMarkovModelMultivariateGaussianDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<IMultivariateDistribution>() { NumberOfStates = NumberOfStatesRightLeft, Delta = Delta });//new HiddenMarkovModelMultivariateGaussianDistribution(NumberOfStatesRightLeft, Delta) { LogNormalized = true };
            model.Normalized = true;
            model.Train(series, NumberOfIterations, LikelihoodTolerance);

            Assert.AreEqual(ModelType.LeftRight, model.Type);
            Assert.AreEqual(NumberOfStatesRightLeft, model.Emission.Length);
            Assert.IsTrue(model.Normalized);
            for (int n = 0; n < NumberOfStatesRightLeft; n++)
            {
                Assert.IsInstanceOfType(model.Emission[n], typeof(NormalDistribution));
            }
        }

        [TestMethod]
        public void Predict_FTSESeriesAndErgodicModel_PredictionResult()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));

            var model = (HiddenMarkovModelMultivariateGaussianDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<IMultivariateDistribution>() { NumberOfStates = NumberOfStates });//new HiddenMarkovModelMultivariateGaussianDistribution(NumberOfStates) { LogNormalized = true };
            model.Normalized = true;
            model.Train(series, NumberOfIterations, LikelihoodTolerance);
            var result = model.Predict(PredictorType.HmmLikelihood, series, null);
 
            Assert.IsNotNull(result);            
        }

        [TestMethod]
        public void Predict_FTSESeriesAndLeftRightModel_PredictionResult()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));

            var model = (HiddenMarkovModelMultivariateGaussianDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<IMultivariateDistribution>() { NumberOfStates = NumberOfStatesRightLeft, Delta = Delta });//new HiddenMarkovModelMultivariateGaussianDistribution(NumberOfStatesRightLeft, Delta) { LogNormalized = true };
            model.Normalized = true;
            model.Train(series, NumberOfIterations, LikelihoodTolerance);
            var result = model.Predict(PredictorType.HmmLikelihood, series, null);

            Assert.IsNotNull(result);  
        }

        [TestMethod]
        public void Predict_FTSESeriesAndErgodicModelAnd20DaysTestSequence_PredictionResult()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var test = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 12, 18), new DateTime(2012, 01, 18));

            var model = (HiddenMarkovModelMultivariateGaussianDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<IMultivariateDistribution>() { NumberOfStates = NumberOfStates });//new HiddenMarkovModelMultivariateGaussianDistribution(NumberOfStates) { LogNormalized = true };
            model.Normalized = true;
            model.Train(series, NumberOfIterations, LikelihoodTolerance);
            var result = model.Predict(PredictorType.HmmLikelihood, series, test, null, 20);

            Assert.IsNotNull(result);
            Assert.AreEqual(20, result.Predicted.Length);
        }

        [TestMethod]
        public void Predict_FTSESeriesAndRightLeftModelAnd20DaysTestSequence_PredictionResult()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var test = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 12, 18), new DateTime(2012, 01, 18));

            var model = (HiddenMarkovModelMultivariateGaussianDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<IMultivariateDistribution>() { NumberOfStates = NumberOfStatesRightLeft, Delta = Delta });//new HiddenMarkovModelMultivariateGaussianDistribution(NumberOfStatesRightLeft, Delta) { LogNormalized = true };
            model.Normalized = true;
            model.Train(series, NumberOfIterations, LikelihoodTolerance);
            var result = model.Predict(PredictorType.HmmLikelihood, series, test, null, 20);

            Assert.IsNotNull(result);
            Assert.AreEqual(20, result.Predicted.Length);
        }

        [TestMethod]
        public void EvaluatePrediction_FTSESeriesAndErgodicModelAnd20DaysTestSequence_PredictionEvaluated()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var test = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 12, 18), new DateTime(2012, 01, 18));

            var model = (HiddenMarkovModelMultivariateGaussianDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<IMultivariateDistribution>() { NumberOfStates = NumberOfStates });//new HiddenMarkovModelMultivariateGaussianDistribution(NumberOfStates) { LogNormalized = true };
            model.Normalized = true;
            model.Train(series, NumberOfIterations, LikelihoodTolerance);
            var result = model.Predict(PredictorType.HmmLikelihood, series, test, null, 20);
            var mape = model.EvaluatePrediction(result, test);

            Assert.IsNotNull(mape);
            Assert.AreEqual(4, mape.MeanAbsolutePercentError.Length);
        }

        [TestMethod]
        public void EvaluatePrediction_FTSESeriesAndRightLeftModelAnd20DaysTestSequence_PredictionEvaluated()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var test = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 12, 18), new DateTime(2012, 01, 18));

            var model = (HiddenMarkovModelMultivariateGaussianDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<IMultivariateDistribution>() { NumberOfStates = NumberOfStatesRightLeft, Delta = Delta });//new HiddenMarkovModelMultivariateGaussianDistribution(NumberOfStatesRightLeft, Delta) { LogNormalized = true };
            model.Normalized = true;
            model.Train(series, NumberOfIterations, LikelihoodTolerance);
            var result = model.Predict(PredictorType.HmmLikelihood, series, test, null, 20);
            var mape = model.EvaluatePrediction(result, test);

            Assert.IsNotNull(mape);
            Assert.AreEqual(4, mape.MeanAbsolutePercentError.Length);
        }
    }
}
