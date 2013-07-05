using System;
using System.Collections.Generic;
using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator.EstimationParameters;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions.Univariate;

namespace HmmDotNet.MachineLearning.Test
{
    [TestClass]
    public class VaribaleEstimatorsTest
    {
        [TestMethod]
        public void BetaEstimator_ABBAObservations_NotNormalizedTest()
        {
            var startDistribution = new[] { 0.85, 0.15 };
            // s = 0, t = 1
            var tpm = new double[2][];
            tpm[0] = new[] { 0.3, 0.7 };
            tpm[1] = new[] { 0.1, 0.9 };

            var observations = new List<IObservation>
                                {
                                    new Observation(new double[] {0}, "A"),
                                    new Observation(new double[] {1}, "B"),
                                    new Observation(new double[] {1}, "B"),
                                    new Observation(new double[] {0}, "A")
                                };

            var emissions = new DiscreteDistribution[2];
            emissions[0] = new DiscreteDistribution(new double[] { 0, 1 }, new[] { 0.4, 0.6 });
            emissions[1] = new DiscreteDistribution(new double[] { 0, 1 }, new[] { 0.5, 0.5 });

            var model = HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<DiscreteDistribution>(){Pi = startDistribution, TransitionProbabilityMatrix = tpm, Emissions = emissions});//new HiddenMarkovModel(startDistribution, tpm, emissions) { LogNormalized = false };
            model.Normalized = false;

            var betaEstimator = new BetaEstimator<DiscreteDistribution>();
            var beta = betaEstimator.Estimate(new BasicEstimationParameters<DiscreteDistribution> { Model = model, Observations = observations, Normalized = model.Normalized });

            Assert.AreEqual(1d, Math.Round(beta[3][0], 9));
            Assert.AreEqual(1d, Math.Round(beta[3][1], 9));
            Assert.AreEqual(0.47, Math.Round(beta[2][0], 9));
            Assert.AreEqual(0.49, Math.Round(beta[2][1], 9));
            Assert.AreEqual(0.2561, Math.Round(beta[1][0], 9));
            Assert.AreEqual(0.2487, Math.Round(beta[1][1], 9));
            Assert.AreEqual(0.133143, Math.Round(beta[0][0], 9));
            Assert.AreEqual(0.127281, Math.Round(beta[0][1], 9));
        }
        [TestMethod]
        public void BetaEstimator_NormalizedTest()
        {
            Assert.Fail("BetaEstimator_NormalizedTest not implemented");
        }
        [TestMethod]
        public void AlphaEstimator_NormalizedTest()
        {
            Assert.Fail("AlphaEstimator_NormalizedTest not implemented");
        }
        [TestMethod]
        public void AlphaEstimator_ABBAObservations_NotNormalizedTest()
        {
            var startDistribution = new[] { 0.85, 0.15 };
            // s = 0, t = 1
            var tpm = new double[2][];
            tpm[0] = new[] { 0.3, 0.7 };
            tpm[1] = new[] { 0.1, 0.9 };

            var observations = new List<IObservation>
                                {
                                    new Observation(new double[] {0}, "A"),
                                    new Observation(new double[] {1}, "B"),
                                    new Observation(new double[] {1}, "B"),
                                    new Observation(new double[] {0}, "A")
                                };

            var emissions = new DiscreteDistribution[2];
            emissions[0] = new DiscreteDistribution(new double[] { 0, 1 }, new[] { 0.4, 0.6 });
            emissions[1] = new DiscreteDistribution(new double[] { 0, 1 }, new[] { 0.5, 0.5 });

            var model = HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<DiscreteDistribution>() { Pi = startDistribution, TransitionProbabilityMatrix = tpm, Emissions = emissions });//new HiddenMarkovModel(startDistribution, tpm, emissions) { LogNormalized = false };
            model.Normalized = false;

            var alphaEstimator = new AlphaEstimator<DiscreteDistribution>();
            var alpha = alphaEstimator.Estimate(new BasicEstimationParameters<DiscreteDistribution> { Model = model, Observations = observations, Normalized = model.Normalized });
            Assert.AreEqual(0.34, Math.Round(alpha[0][0], 9));
            Assert.AreEqual(0.075, Math.Round(alpha[0][1], 9));
            Assert.AreEqual(0.0657, Math.Round(alpha[1][0], 9));
            Assert.AreEqual(0.15275, Math.Round(alpha[1][1], 9));
            Assert.AreEqual(0.020991, Math.Round(alpha[2][0], 9));
            Assert.AreEqual(0.0917325, Math.Round(alpha[2][1], 9));
            Assert.AreEqual(0.00618822, Math.Round(alpha[3][0], 9));
            Assert.AreEqual(0.048626475, Math.Round(alpha[3][1], 9));
        }

        [TestMethod]
        public void GammaEstimator_ABBAObservations_NotNormalizedTest()
        {
            var startDistribution = new[] { 0.85, 0.15 };
            // s = 0, t = 1
            var tpm = new double[2][];
            tpm[0] = new[] { 0.3, 0.7 };
            tpm[1] = new[] { 0.1, 0.9 };

            var observations = new List<IObservation>
                                {
                                    new Observation(new double[] {0}, "A"),
                                    new Observation(new double[] {1}, "B"),
                                    new Observation(new double[] {1}, "B"),
                                    new Observation(new double[] {0}, "A")
                                };

            var emissions = new DiscreteDistribution[2];
            emissions[0] = new DiscreteDistribution(new double[] { 0, 1 }, new[] { 0.4, 0.6 });
            emissions[1] = new DiscreteDistribution(new double[] { 0, 1 }, new[] { 0.5, 0.5 });

            var model = HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<DiscreteDistribution>() { Pi = startDistribution, TransitionProbabilityMatrix = tpm, Emissions = emissions });//new HiddenMarkovModel(startDistribution, tpm, emissions) { LogNormalized = false };
            model.Normalized = false;
            var baseParameters = new BasicEstimationParameters<DiscreteDistribution> { Model = model, Observations = observations, Normalized = model.Normalized };
            var alphaEstimator = new AlphaEstimator<DiscreteDistribution>();
            var alpha = alphaEstimator.Estimate(baseParameters);
            var betaEstimator = new BetaEstimator<DiscreteDistribution>();
            var beta = betaEstimator.Estimate(baseParameters);
            
            var @params = new AdvancedEstimationParameters<DiscreteDistribution>
            {
                Alpha = alpha,
                Beta = beta,
                Observations = observations,
                Model = model
            };
            var gammaEstimator = new GammaEstimator<DiscreteDistribution>();
            var gamma = gammaEstimator.Estimate(@params);
            Assert.AreEqual(0.8258482510939813, gamma[0][0]);
            Assert.AreEqual(0.17415174890601867, gamma[0][1]);
            Assert.AreEqual(1d, gamma[0].Sum());

            Assert.AreEqual(0.3069572858154187, gamma[1][0]);
            Assert.AreEqual(0.69304271418458141, gamma[1][1]);
            Assert.AreEqual(1d, gamma[1].Sum());

            Assert.AreEqual(0.17998403530294202, gamma[2][0]);
            Assert.AreEqual(0.82001596469705806, gamma[2][1]);
            Assert.AreEqual(1d, gamma[2].Sum());

            Assert.AreEqual(0.112893449466425, gamma[3][0]);
            Assert.AreEqual(0.887106550533575, gamma[3][1]);
            Assert.AreEqual(1d, gamma[2].Sum());
        }

        [TestMethod]
        public void GammaEstimator_NormalizedTest()
        {
            Assert.Fail("GammaEstimator_NormalizedTest not implemented");
        }

        [TestMethod]
        public void KsiEstimator_ABBAObservation_NotNormalizedTest()
        {
            var startDistribution = new[] { 0.85, 0.15 };
            // s = 0, t = 1
            var tpm = new double[2][];
            tpm[0] = new[] { 0.3, 0.7 };
            tpm[1] = new[] { 0.1, 0.9 };

            var observations = new List<IObservation>
                                {
                                    new Observation(new double[] {0}, "A"),
                                    new Observation(new double[] {1}, "B"),
                                    new Observation(new double[] {1}, "B"),
                                    new Observation(new double[] {0}, "A")
                                };

            var emissions = new DiscreteDistribution[2];
            emissions[0] = new DiscreteDistribution(new double[] { 0, 1 }, new[] { 0.4, 0.6 });
            emissions[1] = new DiscreteDistribution(new double[] { 0, 1 }, new[] { 0.5, 0.5 });

            var model = HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<DiscreteDistribution>() { Pi = startDistribution, TransitionProbabilityMatrix = tpm, Emissions = emissions });//new HiddenMarkovModel(startDistribution, tpm, emissions) { LogNormalized = false };
            model.Normalized = false;
            var baseParameters = new BasicEstimationParameters<DiscreteDistribution> { Model = model, Observations = observations, Normalized = model.Normalized };
            var alphaEstimator = new AlphaEstimator<DiscreteDistribution>();
            var alpha = alphaEstimator.Estimate(baseParameters);
            var betaEstimator = new BetaEstimator<DiscreteDistribution>();
            var beta = betaEstimator.Estimate(baseParameters);

            var @params = new AdvancedEstimationParameters<DiscreteDistribution>
                {
                    Alpha = alpha,
                    Beta = beta,
                    Observations = observations,
                    Model = model,
                    Normalized = model.Normalized
                };

            var ksiEstimator = new KsiEstimator<DiscreteDistribution>();
            var ksi = ksiEstimator.Estimate(@params);

            Assert.AreEqual(0.28593281418422561, ksi[0][0, 0]);
            Assert.AreEqual(0.53991543690975563, ksi[0][0, 1]);
            Assert.AreEqual(0.021024471631193059, ksi[0][1, 0]);
            Assert.AreEqual(0.15312727727482567, ksi[0][1, 1]);
            Assert.AreEqual(1d, ksi[0].Sum());

            Assert.AreEqual(0.10140018110107153, ksi[1][0, 0]);
            Assert.AreEqual(0.20555710471434716, ksi[1][0, 1]);
            Assert.AreEqual(0.0785838542018705, ksi[1][1, 0]);
            Assert.AreEqual(0.61445885998271088, ksi[1][1, 1]);
            Assert.AreEqual(1d, ksi[1].Sum());

            Assert.AreEqual(0.045953370715644766, ksi[2][0, 0]);
            Assert.AreEqual(0.13403066458729723, ksi[2][0, 1]);
            Assert.AreEqual(0.06694007875078023, ksi[2][1, 0]);
            Assert.AreEqual(0.75307588594627772, ksi[2][1, 1]);
            Assert.AreEqual(1d, ksi[2].Sum());
        }
    }
}
