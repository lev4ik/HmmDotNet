using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Estimators;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Statistics.Distributions.Univariate;

namespace HmmDotNet.MachineLearning.Test
{
    /// <summary>
    /// Summary description for FarwardBackwardTest
    /// </summary>
    [TestClass]
    public class ForwardBackwardTest
    {
        private TestContext testContextInstance;

        /// <summary>
        ///Gets or sets the test context which provides
        ///information about and functionality for the current test run.
        ///</summary>
        public TestContext TestContext
        {
            get
            {
                return testContextInstance;
            }
            set
            {
                testContextInstance = value;
            }
        }

        [TestMethod]
        public void TestForwardRun1()
        {
            var states = new List<IState> {new State(0, "H"), new State(1, "L")};

            var startDistribution = new [] { 0.5, 0.5 };

            var tpm = new double[2][];
            tpm[0] = new[] { 0.5, 0.5 };
            tpm[1] = new[] { 0.4, 0.6 };

            var observations = new List<IObservation>
                                   {
                                       new Observation(new double[] {2}, "G"),
                                       new Observation(new double[] {2}, "G"),
                                       new Observation(new double[] {1}, "C"),
                                       new Observation(new double[] {0}, "A")
                                   };

            var emissions = new DiscreteDistribution[2];
            emissions[0] = new DiscreteDistribution(new double[] { 0, 1, 2, 3 }, new[] { 0.2, 0.3, 0.3, 0.2 });
            emissions[1] = new DiscreteDistribution(new double[] { 0, 1, 2, 3 }, new[] { 0.3, 0.2, 0.2, 0.3 });

            var algo = new ForwardBackward(false);
            var res = algo.RunForward(observations, HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<DiscreteDistribution>() { Pi = startDistribution, TransitionProbabilityMatrix = tpm, Emissions = emissions }));//new HiddenMarkovModelState<DiscreteDistribution>(startDistribution, tpm, emissions));

            Assert.AreEqual(0.0038431500000000005, res);
        }

        [TestMethod]
        public void TestForwardNormalizedRun1()
        {
            var states = new List<IState> { new State(0, "H"), new State(1, "L") };

            var startDistribution = new [] { 0.5, 0.5 };

            var tpm = new double[2][];
            tpm[0] = new[] { 0.5, 0.5 };
            tpm[1] = new[] { 0.4, 0.6 };

            var observations = new List<IObservation>
                                   {
                                       new Observation(new double[] {2}, "G"),
                                       new Observation(new double[] {2}, "G"),
                                       new Observation(new double[] {1}, "C"),
                                       new Observation(new double[] {0}, "A")
                                   };


            var emissions = new DiscreteDistribution[2];
            emissions[0] = new DiscreteDistribution(new double[] { 0, 1, 2, 3 }, new[] { 0.2, 0.3, 0.3, 0.2 });
            emissions[1] = new DiscreteDistribution(new double[] { 0, 1, 2, 3 }, new[] { 0.3, 0.2, 0.2, 0.3 });

            var algo = new ForwardBackward(true);
            var res = algo.RunForward(observations, HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<DiscreteDistribution>() { Pi = startDistribution, TransitionProbabilityMatrix = tpm, Emissions = emissions }));//new HiddenMarkovModelState<DiscreteDistribution>(startDistribution, tpm, emissions));
            // TODO : Check for Log
            Assert.AreEqual(-5.5614629361549142, res);
        }

        [TestMethod]
        public void TestForwardRun2()
        {            
            var states = new List<IState> {new State(0, "s"), new State(1, "t")};

            var startDistribution = new [] { 0.85, 0.15 };

            var tpm = new double[2][];
            tpm[0] = new [] { 0.3, 0.7 };
            tpm[1] = new [] { 0.1, 0.9 };

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
            
            var algo = new ForwardBackward(false);
            var res = algo.RunForward(observations, HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<DiscreteDistribution>() { Pi = startDistribution, TransitionProbabilityMatrix = tpm, Emissions = emissions }));//new HiddenMarkovModel(startDistribution, tpm, emissions));

            Assert.AreEqual(0.05481469500000001, res);
        }

        [TestMethod]
        public void TestForwardNormalizedRun2()
        {
            var states = new List<IState> { new State(0, "s"), new State(1, "t") };

            var startDistribution = new [] { 0.85, 0.15 };

            var tpm = new double[2][];
            tpm[0] = new [] { 0.3, 0.7 };
            tpm[1] = new [] { 0.1, 0.9 };

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

            var algo = new ForwardBackward(true);
            var res = algo.RunForward(observations, HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<DiscreteDistribution>() { Pi = startDistribution, TransitionProbabilityMatrix = tpm, Emissions = emissions }));//new HiddenMarkovModelState<DiscreteDistribution>(startDistribution, tpm, emissions));  
            Assert.AreEqual(-2.9037969640415056, res); 
        }

        [TestMethod]
        public void TestBackwardRun2()
        {
            var states = new List<IState> {new State(0, "s"), new State(1, "t")};

            var startDistribution = new [] { 0.85, 0.15 };

            var tpm = new double[2][];
            tpm[0] = new [] { 0.3, 0.7 };
            tpm[1] = new [] { 0.1, 0.9 };

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
            
            var algo = new ForwardBackward(false);
            var res = algo.RunBackward(observations, HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<DiscreteDistribution>() { Pi = startDistribution, TransitionProbabilityMatrix = tpm, Emissions = emissions }));//new HiddenMarkovModelState<DiscreteDistribution>(startDistribution, tpm, emissions));

            Assert.AreEqual(0.25499 ,res);
            Assert.AreEqual(1, algo.Beta[3][0]);
            Assert.AreEqual(1, algo.Beta[3][1]);
            Assert.AreEqual(0.47, algo.Beta[2][0]);
            Assert.AreEqual(0.49, algo.Beta[2][1]);
            Assert.AreEqual(0.2561, algo.Beta[1][0]);
            Assert.AreEqual(0.2487, algo.Beta[1][1]);
            Assert.AreEqual(0.133143, algo.Beta[0][0]);
            Assert.AreEqual(0.127281, algo.Beta[0][1]);
        }

        [TestMethod]
        public void TestBackwardNormalizedRun2()
        {
            var states = new List<IState> { new State(0, "s"), new State(1, "t") };

            var startDistribution = new [] { 0.85, 0.15 };

            var tpm = new double[2][];
            tpm[0] = new [] { 0.3, 0.7 };
            tpm[1] = new [] { 0.1, 0.9 };

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

            var algo = new ForwardBackward(true);
            var res = algo.RunBackward(observations, HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<DiscreteDistribution>() { Pi = startDistribution, TransitionProbabilityMatrix = tpm, Emissions = emissions }));//new HiddenMarkovModelState<DiscreteDistribution>(startDistribution, tpm, emissions));

            Assert.AreEqual(-1.3665309502789404, res);
            Assert.AreEqual(0d, algo.Beta[3][0]);
            Assert.AreEqual(0d, algo.Beta[3][1]);
            Assert.AreEqual(-0.75502258427803293, algo.Beta[2][0]);
            Assert.AreEqual(-0.71334988787746456, algo.Beta[2][1]);
            Assert.AreEqual(-1.3621872857766575, algo.Beta[1][0]);
            Assert.AreEqual(-1.3915079281727778, algo.Beta[1][1]);
            Assert.AreEqual(-2.0163315403910613, algo.Beta[0][0]);
            Assert.AreEqual(-2.0613580382895655, algo.Beta[0][1]);
        }

    }
}