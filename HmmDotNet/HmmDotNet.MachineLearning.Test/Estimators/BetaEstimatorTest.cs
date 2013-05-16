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
    public class BetaEstimatorTest
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
        public void BetaEstimator_ModelAndObservations_BetaEstimatorCreated()
        {
            const int numberOfStates = 2;

            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = numberOfStates});//new HiddenMarkovModelState<NormalDistribution>(numberOfStates) { LogNormalized = true };
            model.Normalized = true;

            var estimator = new BetaEstimator<NormalDistribution>(model, Helper.Convert(observations), true);
            Assert.IsNotNull(estimator);
        }

        [TestMethod]
        public void Beta_ErgodicLogNormalized_BetaCalculated()
        {
            const int numberOfStates = 2;

            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = numberOfStates, Emissions = CreateEmissions(observations, numberOfStates) });//new HiddenMarkovModelState<NormalDistribution>(numberOfStates, CreateEmissions(observations, numberOfStates)) { LogNormalized = true };
            model.Normalized = true;

            var estimator = new BetaEstimator<NormalDistribution>(model, Helper.Convert(observations), true);
            var beta = estimator.Beta;

            Assert.IsNotNull(beta);
            for (int i = 0; i < observations.Length - 1; i++)
            {
                for (int j = 0; j < numberOfStates; j++)
                {
                    Assert.IsTrue(beta[i][j] < 0, string.Format("Failed Beta [{0}][{1}] : {2}", i, j, beta[i][j]));
                }
            }
            // Last observation has probability == 1
            Assert.IsTrue(beta[observations.Length - 1][0] == 0);
            Assert.IsTrue(beta[observations.Length - 1][1] == 0);
        }

        [TestMethod]
        public void Beta_ErgodicNotNormalized_BetaCalculated()
        {
            const int numberOfStates = 2;

            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = numberOfStates, Emissions = CreateEmissions(observations, numberOfStates) });//new HiddenMarkovModelState<NormalDistribution>(numberOfStates, CreateEmissions(observations, numberOfStates)) { LogNormalized = true };
            model.Normalized = true;

            var estimator = new BetaEstimator<NormalDistribution>(model, Helper.Convert(observations), false);
            var beta = estimator.Beta;

            Assert.IsNotNull(beta);
            for (int i = 0; i < observations.Length - 1; i++)
            {
                for (int j = 0; j < numberOfStates; j++)
                {
                    Assert.IsTrue(beta[i][j] > 0 && beta[i][j] < 1, string.Format("Failed Beta [{0}][{1}] : {2}", i, j, beta[i][j]));
                }
            }
            // Last observation has probability == 1
            Assert.IsTrue(beta[observations.Length - 1][0] == 1);
            Assert.IsTrue(beta[observations.Length - 1][1] == 1);
        }

        [TestMethod]
        public void Beta_RightLeftLogNormalized_BetaCalculated()
        {
            const int numberOfStates = 4;
            const int delta = 3;
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = numberOfStates, Delta = delta, Emissions = CreateEmissions(observations, numberOfStates) });//new HiddenMarkovModelState<NormalDistribution>(numberOfStates, delta, CreateEmissions(observations, numberOfStates)) { LogNormalized = true };
            model.Normalized = true;

            var estimator = new BetaEstimator<NormalDistribution>(model, Helper.Convert(observations), true);
            var beta = estimator.Beta;

            Assert.IsNotNull(beta);
            for (int i = 0; i < observations.Length - 1; i++)
            {
                for (int j = 0; j < numberOfStates; j++)
                {
                    Assert.IsTrue(beta[i][j] < 0, string.Format("Failed Beta [{0}][{1}] : {2}", i, j, beta[i][j]));
                }
            }
            // Last observation has probability == 1
            Assert.IsTrue(beta[observations.Length - 1][0] == 0);
            Assert.IsTrue(beta[observations.Length - 1][1] == 0);
        }

        [TestMethod]
        public void Beta_RightLeftNotNormalized_BetaCalculated()
        {
            const int numberOfStates = 4;
            const int delta = 3;
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = numberOfStates, Delta = delta, Emissions = CreateEmissions(observations, numberOfStates) });//new HiddenMarkovModelState<NormalDistribution>(numberOfStates, delta, CreateEmissions(observations, numberOfStates)) { LogNormalized = true };
            model.Normalized = true;
            var estimator = new BetaEstimator<NormalDistribution>(model, Helper.Convert(observations), false);
            var beta = estimator.Beta;

            Assert.IsNotNull(beta);
            for (int i = 0; i < observations.Length - 1; i++)
            {
                for (int j = 0; j < numberOfStates; j++)
                {
                    Assert.IsTrue(beta[i][j] > 0 && beta[i][j] < 1, string.Format("Failed Beta [{0}][{1}] : {2}", i, j, beta[i][j]));
                }
            }
            // Last observation has probability == 1
            Assert.IsTrue(beta[observations.Length - 1][0] == 1);
            Assert.IsTrue(beta[observations.Length - 1][1] == 1);
        }
    }
}
