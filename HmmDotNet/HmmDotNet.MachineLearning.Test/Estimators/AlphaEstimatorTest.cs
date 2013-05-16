using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Algorithms.Clustering;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.Estimators;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.Logic.Test.MachineLearning.Estimators
{
    [TestClass]
    public class AlphaEstimatorTest
    {
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
        public void AlphaEstimator_ModelAndObservations_AlphaEstimatorCreated()
        {
            const int numberOfStates = 2;

            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = numberOfStates });//new HiddenMarkovModelState<NormalDistribution>(numberOfStates) { LogNormalized = true };
            model.Normalized = true;

            var estimator = new AlphaEstimator<NormalDistribution>(model, Helper.Convert(observations), true);
            Assert.IsNotNull(estimator);
        }

        [TestMethod]
        public void Alpha_ErgodicAndLogNormalized_AlphaCalculated()
        {
            const int numberOfStates = 2;
                
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = numberOfStates, Emissions = CreateEmissions(observations, numberOfStates) });//new HiddenMarkovModelState<NormalDistribution>(numberOfStates, CreateEmissions(observations, numberOfStates)) { LogNormalized = true };
            model.Normalized = true;
            
            var estimator = new AlphaEstimator<NormalDistribution>(model, Helper.Convert(observations), true);
            var alpha = estimator.Alpha;

            Assert.IsNotNull(alpha);
            for (int i = 0; i < observations.Length; i++)
            {
                for (int j = 0; j < numberOfStates; j++)
                {
                    Assert.IsTrue(alpha[i][j] < 0, string.Format("Failed Alpha [{0}][{1}] : {2}", i, j, alpha[i][j]));
                }
            }
        }

        [TestMethod]
        public void Alpha_ErgodicAndNotNormalized_AlphaCalculated()
        {
            const int numberOfStates = 2;

            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = numberOfStates, Emissions = CreateEmissions(observations, numberOfStates) });//new HiddenMarkovModelState<NormalDistribution>(numberOfStates, CreateEmissions(observations, numberOfStates)) { LogNormalized = true };
            model.Normalized = true;

            var estimator = new AlphaEstimator<NormalDistribution>(model, Helper.Convert(observations), false);
            var alpha = estimator.Alpha;

            Assert.IsNotNull(alpha);
            for (int i = 0; i < observations.Length; i++)
            {
                for (int j = 0; j < numberOfStates; j++)
                {
                    Assert.IsTrue(alpha[i][j] > 0 && alpha[i][j] < 1, string.Format("Failed Alpha [{0}][{1}] : {2}", i, j, alpha[i][j]));
                }
            }
        }

        [TestMethod]
        public void Alpha_RightLeftAndLogNormalized_AlphaCalculated()
        {
            const int numberOfStates = 4;
            const int delta = 3;
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = numberOfStates, Delta = delta, Emissions = CreateEmissions(observations, numberOfStates) });//new HiddenMarkovModelState<NormalDistribution>(numberOfStates, delta, CreateEmissions(observations, numberOfStates)) { LogNormalized = true };
            model.Normalized = true;

            var estimator = new AlphaEstimator<NormalDistribution>(model, Helper.Convert(observations), true);
            var alpha = estimator.Alpha;

            Assert.IsNotNull(alpha);
            for (int i = 0; i < observations.Length; i++)
            {
                for (int j = 0; j < numberOfStates; j++)
                {
                    Assert.IsTrue(alpha[i][j] < 0 || double.IsNaN(alpha[i][j]), string.Format("Failed Alpha [{0}][{1}] : {2}", i, j, alpha[i][j]));
                }
            }
        }

        [TestMethod]
        public void Alpha_RightLeftAndNotNormalized_AlphaCalculated()
        {
            const int numberOfStates = 4;
            const int delta = 3;
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = numberOfStates, Delta = delta, Emissions = CreateEmissions(observations, numberOfStates) });//new HiddenMarkovModelState<NormalDistribution>(numberOfStates, delta, CreateEmissions(observations, numberOfStates)) { LogNormalized = true };
            model.Normalized = true;

            var estimator = new AlphaEstimator<NormalDistribution>(model, Helper.Convert(observations), false);
            var alpha = estimator.Alpha;

            Assert.IsNotNull(alpha);
            for (int i = 0; i < observations.Length; i++)
            {
                for (int j = 0; j < numberOfStates; j++)
                {
                    Assert.IsTrue(alpha[i][j] >= 0 && alpha[i][j] < 1, string.Format("Failed Alpha [{0}][{1}] : {2}", i, j, alpha[i][j]));
                }
            }
        }
    }
}
