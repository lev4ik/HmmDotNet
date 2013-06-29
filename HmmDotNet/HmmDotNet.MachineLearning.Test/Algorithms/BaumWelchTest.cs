using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions.Univariate;

namespace HmmDotNet.MachineLearning.Test
{
    [TestClass]
    public class BaumWelchTest
    {
        private const int LikelihoodTolerance = 20;

        [TestMethod]
        public void Run_DefaultModelAndObservagtions_TrainedModel()
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

            var symbols = new List<IObservation> { new Observation(new double[] { 0 }, "A"), new Observation(new double[] { 1 }, "B") };

            var model = HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<DiscreteDistribution>() { Pi = startDistribution, TransitionProbabilityMatrix = tpm, Emissions = emissions });//new HiddenMarkovModel(startDistribution, tpm, emissions) { LogNormalized = false };
            model.Normalized = false;
            var algo = new BaumWelch(observations, model, symbols);
            var res = algo.Run(100, LikelihoodTolerance);

            Assert.AreEqual(0.8258482510939813, res.Pi[0]);
            Assert.AreEqual(0.17415174890601867, res.Pi[1]);

            Assert.AreEqual(0.330050127737348, res.TransitionProbabilityMatrix[0][0]);
            Assert.AreEqual(0.669949872262652, res.TransitionProbabilityMatrix[0][1]);
            Assert.AreEqual(0.098712289730350428, res.TransitionProbabilityMatrix[1][0]);
            Assert.AreEqual(0.90128771026964949, res.TransitionProbabilityMatrix[1][1]);

            Assert.AreEqual(0.65845050146912831, res.Emission[0].ProbabilityMassFunction(0));
            Assert.AreEqual(0.34154949853087169, res.Emission[0].ProbabilityMassFunction(1));
            Assert.AreEqual(0.4122484947955643, res.Emission[1].ProbabilityMassFunction(0));
            Assert.AreEqual(0.58775150520443575, res.Emission[1].ProbabilityMassFunction(1));

            Assert.AreEqual(0.0770055392812707, res.Likelihood);
            Assert.AreEqual(1d, res.Pi.Sum());
            Assert.AreEqual(1d, res.TransitionProbabilityMatrix[0].Sum());           
            Assert.AreEqual(1d, Math.Round(res.TransitionProbabilityMatrix[1].Sum(), 5));
            Assert.AreEqual(1d, res.Emission[0].ProbabilityMassFunction(0) + res.Emission[0].ProbabilityMassFunction(1));
            Assert.AreEqual(1d, res.Emission[1].ProbabilityMassFunction(0) + res.Emission[1].ProbabilityMassFunction(1));
        }

        [TestMethod]
        public void Run_DefaultModelAndObservagtionsAndNormalized_TrainedMode()
        {
            var startDistribution = new[] { 0.85, 0.15 };

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

            var symbols = new List<IObservation> { new Observation(new double[] { 0 }, "A"), new Observation(new double[] { 1 }, "B") };

            var model = HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<DiscreteDistribution>() { Pi = startDistribution, TransitionProbabilityMatrix = tpm, Emissions = emissions });//new HiddenMarkovModel(startDistribution, tpm, emissions) { LogNormalized = true};
            model.Normalized = true;
            var algo = new BaumWelch(observations, model, symbols);
            var res = algo.Run(100, LikelihoodTolerance);

            Assert.AreEqual(0.82585, Math.Round(res.Pi[0], 5));
            Assert.AreEqual(0.17415, Math.Round(res.Pi[1], 5));

            Assert.AreEqual(0.33005, Math.Round(res.TransitionProbabilityMatrix[0][0], 5));
            Assert.AreEqual(0.66995, Math.Round(res.TransitionProbabilityMatrix[0][1], 5));
            Assert.AreEqual(0.09871, Math.Round(res.TransitionProbabilityMatrix[1][0], 5));
            Assert.AreEqual(0.90129, Math.Round(res.TransitionProbabilityMatrix[1][1], 5));

            Assert.AreEqual(0.65845, Math.Round(res.Emission[0].ProbabilityMassFunction(0), 5));
            Assert.AreEqual(0.34155, Math.Round(res.Emission[0].ProbabilityMassFunction(1), 5));
            Assert.AreEqual(0.41225, Math.Round(res.Emission[1].ProbabilityMassFunction(0), 5));
            Assert.AreEqual(0.58775, Math.Round(res.Emission[1].ProbabilityMassFunction(1), 5));

            Assert.AreEqual(-2.5638779209981162, res.Likelihood);
            Assert.AreEqual(1d, Math.Round(res.Pi.Sum(), 5));
            Assert.AreEqual(1d, Math.Round(res.TransitionProbabilityMatrix[0].Sum(), 5));
            Assert.AreEqual(1d, Math.Round(res.TransitionProbabilityMatrix[1].Sum(), 5));
            Assert.AreEqual(1d, Math.Round(res.Emission[0].ProbabilityMassFunction(0) + res.Emission[0].ProbabilityMassFunction(1), 5));
            Assert.AreEqual(1d, Math.Round(res.Emission[1].ProbabilityMassFunction(0) + res.Emission[1].ProbabilityMassFunction(1), 5));
        }
    }
}
