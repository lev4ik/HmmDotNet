using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions.Univariate;
using HmmDotNet.Mathematic;

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

            var betaEstimator = new BetaEstimator<DiscreteDistribution>(model, observations, model.Normalized);
            Assert.AreEqual(1d, Math.Round(betaEstimator.Beta[3][0], 9));
            Assert.AreEqual(1d, Math.Round(betaEstimator.Beta[3][1], 9));
            Assert.AreEqual(0.47, Math.Round(betaEstimator.Beta[2][0], 9));
            Assert.AreEqual(0.49, Math.Round(betaEstimator.Beta[2][1], 9));
            Assert.AreEqual(0.2561, Math.Round(betaEstimator.Beta[1][0], 9));
            Assert.AreEqual(0.2487, Math.Round(betaEstimator.Beta[1][1], 9));
            Assert.AreEqual(0.133143, Math.Round(betaEstimator.Beta[0][0], 9));
            Assert.AreEqual(0.127281, Math.Round(betaEstimator.Beta[0][1], 9));
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

            var alphaEstimator = new AlphaEstimator<DiscreteDistribution>(model, observations, model.Normalized);
            Assert.AreEqual(0.34, Math.Round(alphaEstimator.Alpha[0][0], 9));
            Assert.AreEqual(0.075, Math.Round(alphaEstimator.Alpha[0][1], 9));
            Assert.AreEqual(0.0657, Math.Round(alphaEstimator.Alpha[1][0], 9));
            Assert.AreEqual(0.15275, Math.Round(alphaEstimator.Alpha[1][1], 9));
            Assert.AreEqual(0.020991, Math.Round(alphaEstimator.Alpha[2][0], 9));
            Assert.AreEqual(0.0917325, Math.Round(alphaEstimator.Alpha[2][1], 9));
            Assert.AreEqual(0.00618822, Math.Round(alphaEstimator.Alpha[3][0], 9));
            Assert.AreEqual(0.048626475, Math.Round(alphaEstimator.Alpha[3][1], 9));
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

            var alphaEstimator = new AlphaEstimator<DiscreteDistribution>(model, observations, model.Normalized);
            var betaEstimator = new BetaEstimator<DiscreteDistribution>(model, observations, model.Normalized);
            var parameters = new ParameterEstimations<DiscreteDistribution>(model, observations, alphaEstimator.Alpha, betaEstimator.Beta);
            var gammaEstimator = new GammaEstimator<DiscreteDistribution>(parameters, model.Normalized);

            Assert.AreEqual(0.8258482510939813, gammaEstimator.Gamma[0][0]);
            Assert.AreEqual(0.17415174890601867, gammaEstimator.Gamma[0][1]);
            Assert.AreEqual(1d, gammaEstimator.Gamma[0].Sum());

            Assert.AreEqual(0.3069572858154187, gammaEstimator.Gamma[1][0]);
            Assert.AreEqual(0.69304271418458141, gammaEstimator.Gamma[1][1]);
            Assert.AreEqual(1d, gammaEstimator.Gamma[1].Sum());

            Assert.AreEqual(0.17998403530294202, gammaEstimator.Gamma[2][0]);
            Assert.AreEqual(0.82001596469705806, gammaEstimator.Gamma[2][1]);
            Assert.AreEqual(1d, gammaEstimator.Gamma[2].Sum());

            Assert.AreEqual(0.112893449466425, gammaEstimator.Gamma[3][0]);
            Assert.AreEqual(0.887106550533575, gammaEstimator.Gamma[3][1]);
            Assert.AreEqual(1d, gammaEstimator.Gamma[2].Sum());
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

            var alphaEstimator = new AlphaEstimator<DiscreteDistribution>(model, observations, model.Normalized);
            var betaEstimator = new BetaEstimator<DiscreteDistribution>(model, observations, model.Normalized);
            var parameters = new ParameterEstimations<DiscreteDistribution>(model, observations, alphaEstimator.Alpha, betaEstimator.Beta);
            var ksiEstimator = new KsiEstimator<DiscreteDistribution>(parameters, model.Normalized);

            Assert.AreEqual(0.28593281418422561, ksiEstimator.Ksi[0][0, 0]);
            Assert.AreEqual(0.53991543690975563, ksiEstimator.Ksi[0][0, 1]);
            Assert.AreEqual(0.021024471631193059, ksiEstimator.Ksi[0][1, 0]);
            Assert.AreEqual(0.15312727727482567, ksiEstimator.Ksi[0][1, 1]);
            Assert.AreEqual(1d, ksiEstimator.Ksi[0].Sum());

            Assert.AreEqual(0.10140018110107153, ksiEstimator.Ksi[1][0, 0]);
            Assert.AreEqual(0.20555710471434716, ksiEstimator.Ksi[1][0, 1]);
            Assert.AreEqual(0.0785838542018705, ksiEstimator.Ksi[1][1, 0]);
            Assert.AreEqual(0.61445885998271088, ksiEstimator.Ksi[1][1, 1]);
            Assert.AreEqual(1d, ksiEstimator.Ksi[1].Sum());

            Assert.AreEqual(0.045953370715644766, ksiEstimator.Ksi[2][0, 0]);
            Assert.AreEqual(0.13403066458729723, ksiEstimator.Ksi[2][0, 1]);
            Assert.AreEqual(0.06694007875078023, ksiEstimator.Ksi[2][1, 0]);
            Assert.AreEqual(0.75307588594627772, ksiEstimator.Ksi[2][1, 1]);
            Assert.AreEqual(1d, ksiEstimator.Ksi[2].Sum());
        }
    }
}
