using System;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator.EstimationParameters;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Algorithms.Clustering;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.Estimators;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.Logic.Test.MachineLearning.Estimators
{
    [TestClass]
    public class MuMixtureEstimatorTest
    {
        private const int NumberOfStates = 2;
        private const int NumberOfStatesRightLeft = 4;
        private const int NumberOfComponents = 3;

        private Mixture<IMultivariateDistribution>[] CreateEmissions(double[][] observations, int numberOfStates, int numberOfComponents)
        {
            // Create initial emmissions , TMP and Pi are already created
            var algo = new KMeans();
            algo.CreateClusters(observations, numberOfStates * numberOfComponents, KMeans.KMeansDefaultIterations, (numberOfStates * numberOfComponents > 3) ? InitialClusterSelectionMethod.Furthest : InitialClusterSelectionMethod.Random);

            var emissions = new Mixture<IMultivariateDistribution>[numberOfStates];

            for (int i = 0; i < numberOfStates; i++)
            {
                emissions[i] = new Mixture<IMultivariateDistribution>(numberOfComponents, observations[0].Length);
                for (int j = 0; j < numberOfComponents; j++)
                {
                    var mean = algo.ClusterCenters[j + numberOfComponents * i];
                    var covariance = algo.ClusterCovariances[j + numberOfComponents * i];

                    emissions[i].Components[j] = new NormalDistribution(mean, covariance);
                }
            }

            return emissions;
        }

        [TestMethod]
        public void MixtureMuEstimator_MultivariateNormalDistributionParameters_MixtureMuEstimatorCreated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates, NumberOfComponents) });//new HiddenMarkovModelState<Mixture<IMultivariateDistribution>>(NumberOfStates, CreateEmissions(observations, NumberOfStates, NumberOfComponents)) { LogNormalized = true };
            model.Normalized = true;
            var alphaEstimator = new AlphaEstimator<Mixture<IMultivariateDistribution>>();
            var alpha = alphaEstimator.Estimate(new BasicEstimationParameters<Mixture<IMultivariateDistribution>> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized });
            var betaEstimator = new BetaEstimator<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), true);

            var parameters = new ParameterEstimations<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), alpha, betaEstimator.Beta);
            var mu = new MixtureMuEstimator<Mixture<IMultivariateDistribution>>(parameters);

            Assert.IsNotNull(mu);
        }

        [TestMethod]
        public void Mu_ErgodicAndLogNormalized_MuCalculated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates, NumberOfComponents) });//new HiddenMarkovModelState<Mixture<IMultivariateDistribution>>(NumberOfStates, CreateEmissions(observations, NumberOfStates, NumberOfComponents)) { LogNormalized = true };
            model.Normalized = true;
            var alphaEstimator = new AlphaEstimator<Mixture<IMultivariateDistribution>>();
            var alpha = alphaEstimator.Estimate(new BasicEstimationParameters<Mixture<IMultivariateDistribution>> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized });
            var betaEstimator = new BetaEstimator<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), true);

            var parameters = new ParameterEstimations<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), alpha, betaEstimator.Beta);
            var mu = new MixtureMuEstimator<Mixture<IMultivariateDistribution>>(parameters);

            for (int i = 0; i < NumberOfStates; i++)
            {
                for (int l = 0; l < NumberOfComponents; l++)
                {
                    for (int d = 0; d < observations[0].Length; d++)
                    {
                        Assert.IsTrue(mu.Mu[i, l][d] > 0, string.Format("Failed Mu {0}", mu.Mu[i, l][d]));
                    }                    
                }
            }
        }

        [TestMethod]
        public void Mu_ErgodicAndNotLogNormalized_MuCalculated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates, NumberOfComponents) });//new HiddenMarkovModelState<Mixture<IMultivariateDistribution>>(NumberOfStates, CreateEmissions(observations, NumberOfStates, NumberOfComponents)) { LogNormalized = false };
            model.Normalized = false;
            var alphaEstimator = new AlphaEstimator<Mixture<IMultivariateDistribution>>();
            var alpha = alphaEstimator.Estimate(new BasicEstimationParameters<Mixture<IMultivariateDistribution>> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized });
            var betaEstimator = new BetaEstimator<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), false);

            var parameters = new ParameterEstimations<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), alpha, betaEstimator.Beta);
            var mu = new MixtureMuEstimator<Mixture<IMultivariateDistribution>>(parameters);

            for (int i = 0; i < NumberOfStates; i++)
            {
                for (int l = 0; l < NumberOfComponents; l++)
                {
                    for (int d = 0; d < observations[0].Length; d++)
                    {
                        Assert.IsTrue(mu.Mu[i, l][d] > 0, string.Format("Failed Mu {0}", mu.Mu[i, l][d]));
                    }
                }
            }
        }

        [TestMethod]
        public void Mu_RightLeftAndLogNormalized_MuCalculated()
        {
            var delta = 3;
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfStates = NumberOfStatesRightLeft, Delta = delta, Emissions = CreateEmissions(observations, NumberOfStatesRightLeft, NumberOfComponents) });//new HiddenMarkovModelState<Mixture<IMultivariateDistribution>>(NumberOfStatesRightLeft, delta, CreateEmissions(observations, NumberOfStatesRightLeft, NumberOfComponents)) { LogNormalized = true };
            model.Normalized = true;
            var alphaEstimator = new AlphaEstimator<Mixture<IMultivariateDistribution>>();
            var alpha = alphaEstimator.Estimate(new BasicEstimationParameters<Mixture<IMultivariateDistribution>> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized });
            var betaEstimator = new BetaEstimator<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), true);

            var parameters = new ParameterEstimations<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), alpha, betaEstimator.Beta);
            var mu = new MixtureMuEstimator<Mixture<IMultivariateDistribution>>(parameters);

            for (int i = 0; i < NumberOfStatesRightLeft; i++)
            {
                for (int l = 0; l < NumberOfComponents; l++)
                {
                    for (int d = 0; d < observations[0].Length; d++)
                    {
                        Assert.IsTrue(mu.Mu[i, l][d] > 0, string.Format("Failed Mu {0}", mu.Mu[i, l][d]));
                    }
                }
            }
        }

        [TestMethod]
        public void Mu_RightLeftAndNotLogNormalized_MuCalculated()
        {
            var delta = 3;
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfStates = NumberOfStatesRightLeft, Delta = delta, Emissions = CreateEmissions(observations, NumberOfStatesRightLeft, NumberOfComponents) });//new HiddenMarkovModelState<Mixture<IMultivariateDistribution>>(NumberOfStatesRightLeft, delta, CreateEmissions(observations, NumberOfStatesRightLeft, NumberOfComponents)) { LogNormalized = false };
            model.Normalized = false;
            var alphaEstimator = new AlphaEstimator<Mixture<IMultivariateDistribution>>();
            var alpha = alphaEstimator.Estimate(new BasicEstimationParameters<Mixture<IMultivariateDistribution>> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized });
            var betaEstimator = new BetaEstimator<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), false);

            var parameters = new ParameterEstimations<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), alpha, betaEstimator.Beta);
            var mu = new MixtureMuEstimator<Mixture<IMultivariateDistribution>>(parameters);

            for (int i = 0; i < NumberOfStatesRightLeft; i++)
            {
                for (int l = 0; l < NumberOfComponents; l++)
                {
                    for (int d = 0; d < observations[0].Length; d++)
                    {
                        Assert.IsTrue(mu.Mu[i, l][d] > 0, string.Format("Failed Mu {0}", mu.Mu[i, l][d]));
                    }
                }
            }
        }
    }
}
