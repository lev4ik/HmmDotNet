using System;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator.EstimationParameters;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Algorithms.Clustering;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Mathematic;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;

namespace HmmDotNet.Logic.Test.MachineLearning.Estimators
{
    [TestClass]
    public class MixtureCoefficientEstimatorTest
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
        public void MixtureCoefficientsEstimator_Parameters_MixtureCoefficientsEstimatorCreated()
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
            var coefficients = new MixtureCoefficientsEstimator<Mixture<IMultivariateDistribution>>(parameters);

            Assert.IsNotNull(coefficients);
        }

        [TestMethod]
        public void Coefficients_ErgodicAndLogNormilized_CoefficientsCalculated()
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
            var coefficients = new MixtureCoefficientsEstimator<Mixture<IMultivariateDistribution>>(parameters);

            for (int i = 0; i < NumberOfStates; i++)
            {
                for (int l = 0; l < NumberOfComponents; l++)
                {
                    Assert.IsTrue(coefficients.Coefficients[i][l] < 0, string.Format("Failed Coefficients {0}", coefficients.Coefficients[i][l]));
                }
            }
        }

        [TestMethod]
        public void Coefficients_ErgodicAndNotNormilized_CoefficientsCalculated()
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
            var coefficients = new MixtureCoefficientsEstimator<Mixture<IMultivariateDistribution>>(parameters);

            for (int i = 0; i < NumberOfStates; i++)
            {
                for (int l = 0; l < NumberOfComponents; l++)
                {
                    Assert.IsTrue(coefficients.Coefficients[i][l] > 0 && coefficients.Coefficients[i][l] < 1, string.Format("Failed Coefficients {0}", coefficients.Coefficients[i][l]));
                }
            }
        }

        [TestMethod]
        public void Coefficients_RightLeftAndLogNormilized_CoefficientsCalculated()
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
            var coefficients = new MixtureCoefficientsEstimator<Mixture<IMultivariateDistribution>>(parameters);

            for (int i = 0; i < NumberOfStatesRightLeft; i++)
            {
                for (int l = 0; l < NumberOfComponents; l++)
                {
                    Assert.IsTrue(coefficients.Coefficients[i][l] < 0, string.Format("Failed Coefficients {0}", coefficients.Coefficients[i][l]));
                }
            }
        }

        [TestMethod]
        public void Coefficients_RightLeftAndNotNormilized_CoefficientsCalculated()
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
            var coefficients = new MixtureCoefficientsEstimator<Mixture<IMultivariateDistribution>>(parameters);

            for (int i = 0; i < NumberOfStatesRightLeft; i++)
            {
                for (int l = 0; l < NumberOfComponents; l++)
                {
                    Assert.IsTrue(coefficients.Coefficients[i][l] > 0 && coefficients.Coefficients[i][l] < 1, string.Format("Failed Coefficients {0}", coefficients.Coefficients[i][l]));
                }
            }
        }

        [TestMethod]
        public void Denormalized_NormalizedEstimator_SigmaDenormalized()
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
            var coefficients = new MixtureCoefficientsEstimator<Mixture<IMultivariateDistribution>>(parameters);

            for (int i = 0; i < NumberOfStates; i++)
            {
                for (int l = 0; l < NumberOfComponents; l++)
                {
                    Assert.IsTrue(coefficients.Coefficients[i][l] < 0, string.Format("Failed Coefficients {0}", coefficients.Coefficients[i][l]));
                }
            }
            coefficients.Denormalize();
            for (int i = 0; i < NumberOfStates; i++)
            {
                for (int l = 0; l < NumberOfComponents; l++)
                {
                    Assert.IsTrue(coefficients.Coefficients[i][l] > 0 && coefficients.Coefficients[i][l] < 1, string.Format("Failed Coefficients {0}", coefficients.Coefficients[i][l]));
                }
            }
        }

        [TestMethod]
        public void Coefficients_ErgodicAndNotNormilized_EachEntryMatrixIsSummedToOne()
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
            var coefficients = new MixtureCoefficientsEstimator<Mixture<IMultivariateDistribution>>(parameters);

            for (int i = 0; i < NumberOfStates; i++)
            {
                Assert.AreEqual(1.0d, Math.Round(coefficients.Coefficients[i].Sum(), 5), string.Format("Failed Coefficients {0} at component {1}", new Vector(coefficients.Coefficients[i]), i));
            }            
        }

        [TestMethod]
        public void Coefficients_RightLeftAndNotNormilized_EachEntryMatrixIsSummedToOne()
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
            var coefficients = new MixtureCoefficientsEstimator<Mixture<IMultivariateDistribution>>(parameters);

            for (int i = 0; i < NumberOfStates; i++)
            {
                Assert.AreEqual(1.0d, Math.Round(coefficients.Coefficients[i].Sum(), 5), string.Format("Failed Coefficients {0} at component {1}", new Vector(coefficients.Coefficients[i]), i));
            }  
        }

    }
}
