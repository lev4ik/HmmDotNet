using HmmDotNet.MachineLearning.Algorithms.Search.Base;
using HmmDotNet.MachineLearning.Algorithms.Search.Genetic.PopulationInitialization;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HmmDotNet.MachineLearning.Test.Algorithms.Search
{
    [TestClass]
    public class FromTimeSeriesRandomInitializerTest
    {
        [TestMethod]
        public void Initialize_PopulationSizeZero_EmptyPopulation()
        {
            var init = new FromTimeSeriesRandomInitializer(null);

            var result = init.Initialize<IChromosome<decimal>>(0, 0, 0);

            Assert.AreEqual(result.Count, 0);
        }

        [TestMethod]
        public void Initialize_PopulationSize3_PopulationWithSizeThree()
        {
            var trainigSet = new double[3][];
            trainigSet[0] = new double[] { 0.1, 0.2, 0.3 };
            trainigSet[1] = new double[] { 1.1, 1.2, 1.3 };
            trainigSet[2] = new double[] { 2.1, 2.2, 2.3 };

            var init = new FromTimeSeriesRandomInitializer(trainigSet);

            var result = init.Initialize<decimal>(3, 1, 0.1m);

            Assert.AreEqual(result.Count, 3);
        }
    }
}
