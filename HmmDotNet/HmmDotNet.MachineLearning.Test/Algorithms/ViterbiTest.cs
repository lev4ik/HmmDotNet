using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.MachineLearning;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Examples;

namespace HmmDotNet.Logic.Test.MachineLearning.Algorithms
{
    /// <summary>
    /// Summary description for ViterbiTest
    /// </summary>
    [TestClass]
    public class ViterbiTest
    {
        public ViterbiTest()
        {
            //
            // TODO: Add constructor logic here
            //
        }

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
        public void Run_3ObservationsAnd2StatesAndRainySunnyModel_PathCount3()
        {
            var algo = new Viterbi(false);

            var observations = new List<IObservation>
                                   {
                                       new Observation(new double[] {0}, "walk"),
                                       new Observation(new double[] {1}, "shop"),
                                       new Observation(new double[] {2}, "clean")
                                   };

            var startDistribution = new [] { 0.6, 0.4 };
            
            var states = new List<IState> {new State(0, "Rainy"), new State(1, "Sunny")};

            var tpm = new double[][] {new[] {0.7, 0.3}, new[] {0.4, 0.6}};

            var distributions = new List<IDistribution> {new RainyDistribution(), new SunnyDistribution()};

            var path = algo.Run(observations, states, startDistribution, tpm, distributions);

            Assert.AreEqual(path.Count, 3);
            Assert.AreEqual("Sunny", path[0].Description);
            Assert.AreEqual("Rainy", path[1].Description);
            Assert.AreEqual("Rainy", path[2].Description);
        }

        [TestMethod]
        public void Run_3ObservationsAnd2StatesAnd123Model_PathCount3()
        {
            var algo = new Viterbi(false);

            var observations = new List<IObservation>
                                   {
                                       new Observation(new double[] {0}, "Red"),
                                       new Observation(new double[] {1}, "Blue"),
                                       new Observation(new double[] {0}, "Red")
                                   };

            var startDistribution = new [] { 1/3d, 1/3d, 1/3d };

            var states = new List<IState> {new State(0, "One"), new State(1, "Two"), new State(2, "Three")};

            var tpm = new double[][]{new[] {0.3, 0.6, 0.1}, new[] {0.5, 0.2, 0.3}, new[] {0.4, 0.1, 0.5}};

            var distributions = new List<IDistribution>
                                    {new FirstDistribution(), new SecondDistribution(), new ThirdDistribution()};

            var path = algo.Run(observations, states, startDistribution, tpm, distributions);

            Assert.AreEqual(path.Count, 3);
            Assert.AreEqual("One", path[0].Description);
            Assert.AreEqual("Two", path[1].Description);
            Assert.AreEqual("One", path[2].Description);
        }

        [TestMethod]
        public void Run_9ObservationsAnd2StatesAndHLModel_PathCount9()
        {
            var algo = new Viterbi(false);

            var observations = new List<IObservation>
                                   {
                                       new Observation(new double[] {2}, "G"),
                                       new Observation(new double[] {2}, "G"),
                                       new Observation(new double[] {1}, "C"),
                                       new Observation(new double[] {0}, "A"),
                                       new Observation(new double[] {1}, "C"),
                                       new Observation(new double[] {3}, "T"),
                                       new Observation(new double[] {2}, "G"),
                                       new Observation(new double[] {0}, "A"),
                                       new Observation(new double[] {0}, "A")
                                   };

            var startDistribution = new[] { 0.5, 0.5 };

            var states = new List<IState> { new State(0, "H"), new State(1, "L") };

            var tpm = new double[][] { new[] { 0.5, 0.5 } , new[] { 0.4, 0.6 } };

            var distributions = new List<IDistribution> { new HDistribution(), new LDistribution() };

            var path = algo.Run(observations, states, startDistribution, tpm, distributions);

            Assert.AreEqual(path.Count, 9);
            Assert.AreEqual("H", path[0].Description);
            Assert.AreEqual("H", path[1].Description);
            Assert.AreEqual("H", path[2].Description);
            Assert.AreEqual("L", path[3].Description);
            Assert.AreEqual("L", path[4].Description);
            Assert.AreEqual("L", path[5].Description);
            Assert.AreEqual("L", path[6].Description);
            Assert.AreEqual("L", path[7].Description);
            Assert.AreEqual("L", path[8].Description);
        }

        [TestMethod]
        public void Run_5ObservationsAnd3StatesAndHealthySickModel_PathCount5()
        {
            var algo = new Viterbi(false);

            var observations = new List<IObservation>
                                   {
                                       new Observation(new double[] {0}, "high"),
                                       new Observation(new double[] {1}, "average"),
                                       new Observation(new double[] {1}, "average"),
                                       new Observation(new double[] {2}, "low"),
                                       new Observation(new double[] {2}, "low")
                                   };

            var startDistribution = new[] { 1/3d, 1/3d, 1/3d };

            var states = new List<IState> { new State(0, "Healthy"), new State(1, "OK"), new State(2, "Sick") };

            var tpm = new double[][] { new[] { 0.4, 0.3, 0.3 }, new[] { 0.2, 0.6, 0.2 }, new[] { 0, 0.4, 0.6 } };

            var distributions = new List<IDistribution> { new HealthyDistribution(), new OkDistribution(), new SickDistribution() };

            var path = algo.Run(observations, states, startDistribution, tpm, distributions);

            Assert.AreEqual(path.Count, 5);
            Assert.AreEqual("Healthy", path[0].Description);
            Assert.AreEqual("OK", path[1].Description);
            Assert.AreEqual("OK", path[2].Description);
            Assert.AreEqual("Sick", path[3].Description);
            Assert.AreEqual("Sick", path[4].Description);
        }

        [TestMethod]
        public void Run_Normalized3ObservationsAnd3StatesAnd123Model_PathCount3()
        {
            var algo = new Viterbi(true);

            var observations = new List<IObservation>
                                   {
                                       new Observation(new double[] {0}, "Red"),
                                       new Observation(new double[] {1}, "Blue"),
                                       new Observation(new double[] {0}, "Red")
                                   };

            var startDistribution = new[] { 1 / 3d, 1 / 3d, 1 / 3d };

            var states = new List<IState> { new State(0, "One"), new State(1, "Two"), new State(2, "Three") };

            var tpm = new double[][] { new[] { 0.3, 0.6, 0.1 } ,new[] { 0.5, 0.2, 0.3 } ,new[] { 0.4, 0.1, 0.5 } };

            var distributions = new List<IDistribution> { new FirstDistribution(), new SecondDistribution(), new ThirdDistribution() };

            var path = algo.Run(observations, states, startDistribution, tpm, distributions);

            Assert.AreEqual(path.Count, 3);
            Assert.AreEqual("One", path[0].Description);
            Assert.AreEqual("Two", path[1].Description);
            Assert.AreEqual("One", path[2].Description);
        }

        [TestMethod]
        public void Run_Normalized3ObservationsAnd2StatesAndRainySunnyModel_PathCount3()
        {
            var algo = new Viterbi(true);

            var observations = new List<IObservation>
                                   {
                                       new Observation(new double[] {0}, "walk"),
                                       new Observation(new double[] {1}, "shop"),
                                       new Observation(new double[] {2}, "clean")
                                   };

            var startDistribution = new[] { 0.6, 0.4 };

            var states = new List<IState> { new State(0, "Rainy"), new State(1, "Sunny") };

            var tpm = new double[][] { new[] { 0.7, 0.3 } , new[] { 0.4, 0.6 } };

            var distributions = new List<IDistribution> { new RainyDistribution(), new SunnyDistribution() };

            var path = algo.Run(observations, states, startDistribution, tpm, distributions);

            Assert.AreEqual(path.Count, 3);
            Assert.AreEqual("Sunny", path[0].Description);
            Assert.AreEqual("Rainy", path[1].Description);
            Assert.AreEqual("Rainy", path[2].Description);
        }


        [TestMethod]
        public void Run_Normalized9ObservationsAnd2StatesAndHLModel_PathCount9()
        {
            var algo = new Viterbi(true);

            var observations = new List<IObservation>
                                   {
                                       new Observation(new double[] {2}, "G"),
                                       new Observation(new double[] {2}, "G"),
                                       new Observation(new double[] {1}, "C"),
                                       new Observation(new double[] {0}, "A"),
                                       new Observation(new double[] {1}, "C"),
                                       new Observation(new double[] {3}, "T"),
                                       new Observation(new double[] {2}, "G"),
                                       new Observation(new double[] {0}, "A"),
                                       new Observation(new double[] {0}, "A")
                                   };

            var startDistribution = new[] { 0.5, 0.5 };

            var states = new List<IState> { new State(0, "H"), new State(1, "L") };

            var tpm = new double[][] { new[] { 0.5, 0.5 } , new[] { 0.4, 0.6 } };

            var distributions = new List<IDistribution> { new HDistribution(), new LDistribution() };

            var path = algo.Run(observations, states, startDistribution, tpm, distributions);

            Assert.AreEqual(path.Count, 9);
            Assert.AreEqual("H", path[0].Description);
            Assert.AreEqual("H", path[1].Description);
            Assert.AreEqual("H", path[2].Description);
            Assert.AreEqual("L", path[3].Description);
            Assert.AreEqual("L", path[4].Description);
            Assert.AreEqual("L", path[5].Description);
            Assert.AreEqual("L", path[6].Description);
            Assert.AreEqual("L", path[7].Description);
            Assert.AreEqual("L", path[8].Description);
        }

        [TestMethod]
        public void Run_Normalized5ObservationsAnd3StatesAndHealthySickModel_PathCount5()
        {
            var algo = new Viterbi(true);

            var observations = new List<IObservation>
                                   {
                                       new Observation(new double[] {0}, "high"),
                                       new Observation(new double[] {1}, "average"),
                                       new Observation(new double[] {1}, "average"),
                                       new Observation(new double[] {2}, "low"),
                                       new Observation(new double[] {2}, "low")
                                   };

            var startDistribution = new[] { 1 / 3d, 1 / 3d, 1 / 3d };

            var states = new List<IState> { new State(0, "Healthy"), new State(1, "OK"), new State(2, "Sick") };

            var tpm = new double[][] { new[] { 0.4, 0.3, 0.3 }, new[] { 0.2, 0.6, 0.2 }, new[] { 0, 0.4, 0.6 } };

            var distributions = new List<IDistribution> { new HealthyDistribution(), new OkDistribution(), new SickDistribution() };

            var path = algo.Run(observations, states, startDistribution, tpm, distributions);

            Assert.AreEqual(path.Count, 5);
            Assert.AreEqual("Healthy", path[0].Description);
            Assert.AreEqual("OK", path[1].Description);
            Assert.AreEqual("OK", path[2].Description);
            Assert.AreEqual("Sick", path[3].Description);
            Assert.AreEqual("Sick", path[4].Description);
        }
        
        [TestMethod]
        public void Run_4ObservationsAnd2StatesAndSTModel_PathCount4()
        {
            var observations = new List<IObservation>
                                   {
                                       new Observation(new double[] {0}, "A"),
                                       new Observation(new double[] {1}, "B"),
                                       new Observation(new double[] {1}, "B"),
                                       new Observation(new double[] {0}, "A")
                                   };

            var startDistribution = new[] { 0.85, 0.15 };

            var states = new List<IState> { new State(0, "s"), new State(1, "t") };

            var tpm = new double[2][];
            tpm[0] = new[] { 0.3, 0.7 };
            tpm[1] = new[] { 0.1, 0.9 };

            var distributions = new List<IDistribution> { new HealthyDistribution(), new OkDistribution(), new SickDistribution() };

            var algo = new Viterbi(false);
            var path = algo.Run(observations, states, startDistribution, tpm, distributions);

            Assert.AreEqual(path.Count, 4);
            Assert.AreEqual("s", path[0].Description);
            Assert.AreEqual("t", path[1].Description);
            Assert.AreEqual("t", path[2].Description);
            Assert.AreEqual("t", path[3].Description);            
        }
    }
}
