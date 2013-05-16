using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Statistics.Distributions.Univariate;

namespace HmmDotNet.MachineLearning.Test
{
    /// <summary>
    /// Summary description for HMMTest
    /// </summary>
    [TestClass]
    public class HiddenMarkovModelTest
    {        
        [TestMethod]
        public void CreateHiddenMarkovModelTest()
        {
            var n = 3;
            var m = 3;
            var startProbabilityVector = new double[] { 0.5, 0.5, 0 };
            var tpm = new double[n][];
            tpm[0] = new double[]{ 0, 1 / 3, 2 / 3 };
            tpm[1] = new double[]{ 1 / 3, 0, 2 / 3 };
            tpm[2] = new double[]{ 1 / 3, 1 / 3, 1 / 3 };

            var emissions = new DiscreteDistribution[n];
            emissions[0] = new DiscreteDistribution(new double[] { 1, 2, 3 }, new[] { 0.1, 0.4, 0.5 });
            emissions[1] = new DiscreteDistribution(new double[] { 1, 2, 3 }, new[] { 0.6, 0.3, 0.1 });
            emissions[2] = new DiscreteDistribution(new double[] { 1, 2, 3 }, new[] { 0.6, 0.3, 0.1 });
            var symbols = new List<Observation>();
            symbols.Add(new Observation(new double[] {1}, "1"));
            symbols.Add(new Observation(new double[] {2}, "2"));
            symbols.Add(new Observation(new double[] {3}, "3"));

            var hmm = HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<DiscreteDistribution>() { Pi = startProbabilityVector, TransitionProbabilityMatrix = tpm, Emissions = emissions });//new HiddenMarkovModel(startProbabilityVector, tpm, emissions);

            Assert.AreEqual(hmm.N, n);
            Assert.AreEqual(hmm.TransitionProbabilityMatrix.Length, n);
            Assert.AreEqual(hmm.Emission.Length, n);
            Assert.AreEqual(hmm.Pi.Length, m);
            
            //Assert.AreEqual(hmm.NumberOfObservations, m);
            //Assert.AreEqual(hmm.Symbols.Count, m);
        }
    }
}
