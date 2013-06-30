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
using HmmDotNet.Mathematic;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.Logic.Test.MachineLearning.Estimators
{
    [TestClass]
    public class MixtureGammaEstimatorTest
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
        public void MixtureGammaEstimator_ParametersAndLogNormilized_MixtureGammaEstimatorCreated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates, NumberOfComponents) });//new HiddenMarkovModelState<Mixture<IMultivariateDistribution>>(NumberOfStates, CreateEmissions(observations, NumberOfStates, NumberOfComponents)) { LogNormalized = true };
            model.Normalized = true;
            var alphaEstimator = new AlphaEstimator<Mixture<IMultivariateDistribution>>();
            var alpha = alphaEstimator.Estimate(new BasicEstimationParameters<Mixture<IMultivariateDistribution>> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized });
            var betaEstimator = new BetaEstimator<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), true);

            var parameters = new ParameterEstimations<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), alpha, betaEstimator.Beta);
            var gamma = new MixtureGammaEstimator<Mixture<IMultivariateDistribution>>(parameters);

            Assert.IsNotNull(gamma);
        }

        [TestMethod]
        public void MixtureGammaEstimator_Parameters_MixtureGammaComponentsAndGammaInitialized()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates, NumberOfComponents) });//new HiddenMarkovModelState<Mixture<IMultivariateDistribution>>(NumberOfStates, CreateEmissions(observations, NumberOfStates, NumberOfComponents)) { LogNormalized = true };
            model.Normalized = true;
            var alphaEstimator = new AlphaEstimator<Mixture<IMultivariateDistribution>>();
            var alpha = alphaEstimator.Estimate(new BasicEstimationParameters<Mixture<IMultivariateDistribution>> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized });
            var betaEstimator = new BetaEstimator<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), true);

            var parameters = new ParameterEstimations<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), alpha, betaEstimator.Beta);
            var gamma = new MixtureGammaEstimator<Mixture<IMultivariateDistribution>>(parameters);

            Assert.IsNotNull(gamma.Gamma);
            Assert.IsNotNull(gamma.GammaComponents);
        }

        [TestMethod]
        public void GammaComponents_ErgodicAndLogNormilized_GammaComponentsCalculated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates, NumberOfComponents) });//new HiddenMarkovModelState<Mixture<IMultivariateDistribution>>(NumberOfStates, CreateEmissions(observations, NumberOfStates, NumberOfComponents)) { LogNormalized = true };
            model.Normalized = true;
            var alphaEstimator = new AlphaEstimator<Mixture<IMultivariateDistribution>>();
            var alpha = alphaEstimator.Estimate(new BasicEstimationParameters<Mixture<IMultivariateDistribution>> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized });
            var betaEstimator = new BetaEstimator<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), true);

            var parameters = new ParameterEstimations<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), alpha, betaEstimator.Beta);
            var estimator = new MixtureGammaEstimator<Mixture<IMultivariateDistribution>>(parameters);

            Assert.IsNotNull(estimator);
            var gammaComponents = estimator.GammaComponents;
            for (int t = 0; t < observations.Length; t++)
            {
                for (int i = 0; i < NumberOfStates; i++)
                {
                    for (int l = 0; l < NumberOfComponents; l++)
                    {
                        Assert.IsTrue(gammaComponents[t][i, l] < 0, string.Format("Failed Gamma Components {0}", gammaComponents[t][i, l]));
                    }   
                }
            }
        }

        [TestMethod]
        public void GammaComponents_ErgodicNotNormalized_GammaComponentsCalculated()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates, NumberOfComponents) });//new HiddenMarkovModelState<Mixture<IMultivariateDistribution>>(NumberOfStates, CreateEmissions(observations, NumberOfStates, NumberOfComponents)) { LogNormalized = false };
            model.Normalized = false;
            var alphaEstimator = new AlphaEstimator<Mixture<IMultivariateDistribution>>();
            var alpha = alphaEstimator.Estimate(new BasicEstimationParameters<Mixture<IMultivariateDistribution>> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized });
            var betaEstimator = new BetaEstimator<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), false);

            var parameters = new ParameterEstimations<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), alpha, betaEstimator.Beta);
            var estimator = new MixtureGammaEstimator<Mixture<IMultivariateDistribution>>(parameters);

            Assert.IsNotNull(estimator);
            var gammaComponents = estimator.GammaComponents;
            for (int t = 0; t < observations.Length; t++)
            {
                for (int i = 0; i < NumberOfStates; i++)
                {
                    for (int l = 0; l < NumberOfComponents; l++)
                    {
                        Assert.IsTrue(gammaComponents[t][i, l] > 0 && gammaComponents[t][i, l] < 1, string.Format("Failed Gamma Components {0}", gammaComponents[t][i, l]));
                    }
                }
            }
        }

        [TestMethod]
        public void GammaComponents_RightLeftAndLogNormilized_GammaComponentsCalculated()
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
            var estimator = new MixtureGammaEstimator<Mixture<IMultivariateDistribution>>(parameters);

            Assert.IsNotNull(estimator);
            var gammaComponents = estimator.GammaComponents;
            for (int t = 0; t < observations.Length; t++)
            {
                for (int i = 0; i < NumberOfStatesRightLeft; i++)
                {
                    for (int l = 0; l < NumberOfComponents; l++)
                    {
                        Assert.IsTrue(gammaComponents[t][i, l] < 0 || double.IsNaN(gammaComponents[t][i, l]), string.Format("Failed Gamma Components {0}, [{1}][{2},{3}]", gammaComponents[t][i, l], t, i, l));
                    }
                }
            }
        }

        [TestMethod]
        public void GammaComponents_RightLeftAndNotNormalized_GammaComponentsCalculated()
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
            var estimator = new MixtureGammaEstimator<Mixture<IMultivariateDistribution>>(parameters);

            Assert.IsNotNull(estimator);
            var gammaComponents = estimator.GammaComponents;
            for (int t = 0; t < observations.Length; t++)
            {
                for (int i = 0; i < NumberOfStatesRightLeft; i++)
                {
                    for (int l = 0; l < NumberOfComponents; l++)
                    {
                        Assert.IsTrue(gammaComponents[t][i, l] >= 0 && gammaComponents[t][i, l] < 1, string.Format("Failed Gamma Components {0}, [{1}][{2},{3}]", gammaComponents[t][i, l], t, i, l));
                    }
                }
            }
        }

        [TestMethod]
        public void GammaComponents_ErgodicNotNormalized_EachEntryMatrixIsSummedToOne()
        {
            var util = new TestDataUtils();
            var observations = util.GetSvcData(util.FTSEFilePath, new DateTime(2011, 11, 18), new DateTime(2011, 12, 18));
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfStates = NumberOfStates, Emissions = CreateEmissions(observations, NumberOfStates, NumberOfComponents) });//new HiddenMarkovModelState<Mixture<IMultivariateDistribution>>(NumberOfStates, CreateEmissions(observations, NumberOfStates, NumberOfComponents)) { LogNormalized = false };
            model.Normalized = false;
            var alphaEstimator = new AlphaEstimator<Mixture<IMultivariateDistribution>>();
            var alpha = alphaEstimator.Estimate(new BasicEstimationParameters<Mixture<IMultivariateDistribution>> { Model = model, Observations = Helper.Convert(observations), Normalized = model.Normalized });
            var betaEstimator = new BetaEstimator<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), false);

            var parameters = new ParameterEstimations<Mixture<IMultivariateDistribution>>(model, Helper.Convert(observations), alpha, betaEstimator.Beta);
            var estimator = new MixtureGammaEstimator<Mixture<IMultivariateDistribution>>(parameters);

            var gammaComponents = estimator.GammaComponents;
            for (int t = 0; t < observations.Length; t++)
            {
                Assert.AreEqual(1.0d, Math.Round(gammaComponents[t].Sum(),5), string.Format("Failed Gamma Components {0} at time {1}", new Matrix(gammaComponents[t]), t));
            }
            
        }

        [TestMethod]
        public void GammaComponents_RightLeftNotNormalized_EachEntryMatrixIsSummedToOne()
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
            var estimator = new MixtureGammaEstimator<Mixture<IMultivariateDistribution>>(parameters);

            var gammaComponents = estimator.GammaComponents;
            for (int t = 0; t < observations.Length; t++)
            {
                Assert.AreEqual(1.0d, Math.Round(gammaComponents[t].Sum(), 5), string.Format("Failed Gamma Components {0} at time {1}", new Matrix(gammaComponents[t]), t));
            }
        }

    }
}
