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
    public class SigmaMixtureEstimatorTest
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
        public void MixtureSigmaEstimator_Parameters_MixtureSigmaEstimatorCreated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates, NumberOfComponents) });//new HiddenMarkovModelState<Mixture<IMultivariateDistribution>>(NumberOfStates, CreateEmissions(observations, NumberOfStates, NumberOfComponents)) { LogNormalized = true };
            model.Normalized = true;
            var baseParameters = new BasicEstimationParameters<Mixture<IMultivariateDistribution>> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized };
            var alphaEstimator = new AlphaEstimator<Mixture<IMultivariateDistribution>>();
            var alpha = alphaEstimator.Estimate(baseParameters);
            var betaEstimator = new BetaEstimator<Mixture<IMultivariateDistribution>>();
            var beta = betaEstimator.Estimate(baseParameters);

            var parameters = new ParameterEstimations<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), alpha, beta);
            var sigma = new MixtureMuEstimator<Mixture<IMultivariateDistribution>>(parameters);

            Assert.IsNotNull(sigma);
        }

        [TestMethod]
        public void Sigma_ErgodicAndParametersAndLogNormalized_SigmaCalculated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates, NumberOfComponents) });//new HiddenMarkovModelState<Mixture<IMultivariateDistribution>>(NumberOfStates, CreateEmissions(observations, NumberOfStates, NumberOfComponents)) { LogNormalized = true };
            model.Normalized = true;
            var baseParameters = new BasicEstimationParameters<Mixture<IMultivariateDistribution>> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized };
            var alphaEstimator = new AlphaEstimator<Mixture<IMultivariateDistribution>>();
            var alpha = alphaEstimator.Estimate(baseParameters);
            var betaEstimator = new BetaEstimator<Mixture<IMultivariateDistribution>>();
            var beta = betaEstimator.Estimate(baseParameters);

            var parameters = new ParameterEstimations<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), alpha, beta);
            var sigma = new MixtureSigmaEstimator<Mixture<IMultivariateDistribution>>(parameters);

            for (int i = 0; i < NumberOfStates; i++)
            {
                for (int l = 0; l < NumberOfComponents; l++)
                {
                    for (int rows = 0; rows < parameters.Observation[0].Dimention; rows++)
                    {
                        for (int cols = 0; cols < parameters.Observation[0].Dimention; cols++)
                        {
                            Assert.IsTrue(sigma.Sigma[i, l][rows, cols] > 0, string.Format("Failed Sigma {0}", sigma.Sigma[i, l][rows, cols]));
                        }
                    }
                }
            }
        }
        
        [TestMethod]
        public void Sigma_ErgodicAndParametersAndNotNormalized_SigmaCalculated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates, NumberOfComponents) });//new HiddenMarkovModelState<Mixture<IMultivariateDistribution>>(NumberOfStates, CreateEmissions(observations, NumberOfStates, NumberOfComponents)) { LogNormalized = false };
            model.Normalized = false;
            var baseParameters = new BasicEstimationParameters<Mixture<IMultivariateDistribution>> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized };
            var alphaEstimator = new AlphaEstimator<Mixture<IMultivariateDistribution>>();
            var alpha = alphaEstimator.Estimate(baseParameters);
            var betaEstimator = new BetaEstimator<Mixture<IMultivariateDistribution>>();
            var beta = betaEstimator.Estimate(baseParameters);

            var parameters = new ParameterEstimations<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), alpha, beta);
            var sigma = new MixtureSigmaEstimator<Mixture<IMultivariateDistribution>>(parameters);

            for (int i = 0; i < NumberOfStates; i++)
            {
                for (int l = 0; l < NumberOfComponents; l++)
                {
                    for (int rows = 0; rows < parameters.Observation[0].Dimention; rows++)
                    {
                        for (int cols = 0; cols < parameters.Observation[0].Dimention; cols++)
                        {
                            Assert.IsTrue(sigma.Sigma[i, l][rows, cols] > 0, string.Format("Failed Sigma {0}", sigma.Sigma[i, l][rows, cols]));
                        }
                    }
                }
            }
        }
        
        [TestMethod]
        public void Sigma_RightLeftAndParametersAndLogNormalized_SigmaCalculated()
        {
            var delta = 3;
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfStates = NumberOfStatesRightLeft, Delta = delta, Emissions = CreateEmissions(observations, NumberOfStatesRightLeft, NumberOfComponents) });//new HiddenMarkovModelState<Mixture<IMultivariateDistribution>>(NumberOfStatesRightLeft, delta, CreateEmissions(observations, NumberOfStatesRightLeft, NumberOfComponents)) { LogNormalized = true };
            model.Normalized = true;
            var baseParameters = new BasicEstimationParameters<Mixture<IMultivariateDistribution>> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized };
            var alphaEstimator = new AlphaEstimator<Mixture<IMultivariateDistribution>>();
            var alpha = alphaEstimator.Estimate(baseParameters);
            var betaEstimator = new BetaEstimator<Mixture<IMultivariateDistribution>>();
            var beta = betaEstimator.Estimate(baseParameters);

            var parameters = new ParameterEstimations<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), alpha, beta);
            var sigma = new MixtureSigmaEstimator<Mixture<IMultivariateDistribution>>(parameters);

            for (int i = 0; i < NumberOfStatesRightLeft; i++)
            {
                for (int l = 0; l < NumberOfComponents; l++)
                {
                    for (int rows = 0; rows < parameters.Observation[0].Dimention; rows++)
                    {
                        for (int cols = 0; cols < parameters.Observation[0].Dimention; cols++)
                        {
                            Assert.IsTrue(sigma.Sigma[i, l][rows, cols] > 0, string.Format("Failed Sigma {0}", sigma.Sigma[i, l][rows, cols]));
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void Sigma_RightLeftAndParametersAnnNotNormalized_SigmaCalculated()
        {
            var delta = 3;
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfStates = NumberOfStatesRightLeft, Delta = delta, Emissions = CreateEmissions(observations, NumberOfStatesRightLeft, NumberOfComponents) });//new HiddenMarkovModelState<Mixture<IMultivariateDistribution>>(NumberOfStatesRightLeft, delta, CreateEmissions(observations, NumberOfStatesRightLeft, NumberOfComponents)) { LogNormalized = false };
            model.Normalized = false;
            var baseParameters = new BasicEstimationParameters<Mixture<IMultivariateDistribution>> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized };
            var alphaEstimator = new AlphaEstimator<Mixture<IMultivariateDistribution>>();
            var alpha = alphaEstimator.Estimate(baseParameters);
            var betaEstimator = new BetaEstimator<Mixture<IMultivariateDistribution>>();
            var beta = betaEstimator.Estimate(baseParameters);

            var parameters = new ParameterEstimations<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), alpha, beta);
            var sigma = new MixtureSigmaEstimator<Mixture<IMultivariateDistribution>>(parameters);

            for (int i = 0; i < NumberOfStatesRightLeft; i++)
            {
                for (int l = 0; l < NumberOfComponents; l++)
                {
                    for (int rows = 0; rows < parameters.Observation[0].Dimention; rows++)
                    {
                        for (int cols = 0; cols < parameters.Observation[0].Dimention; cols++)
                        {
                            Assert.IsTrue(sigma.Sigma[i, l][rows, cols] > 0, string.Format("Failed Sigma {0}", sigma.Sigma[i, l][rows, cols]));
                        }
                    }
                }
            }
        }

    }
}
