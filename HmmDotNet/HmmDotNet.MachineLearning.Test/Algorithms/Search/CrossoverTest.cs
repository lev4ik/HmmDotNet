using HmmDotNet.MachineLearning.Algorithms.Search;
using HmmDotNet.MachineLearning.Algorithms.Search.Base;
using HmmDotNet.MachineLearning.Algorithms.Search.Genetic;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HmmDotNet.MachineLearning.Test.Algorithms.Search
{
    [TestClass]
    public class CrossoverTest
    {
        [TestMethod]
        public void Crossover_TwoChromosomesTenPercentProbability_CrossoverHappend()
        {
            var probability = 0.1m;
            var representation1 = new IGene<decimal>[2];
            var representation2 = new IGene<decimal>[2];

            var gene1Representation = new[] { 1.2m, 1.3m, 1.4m, 1.5m };
            var gene2Representation = new[] { 1.6m, 1.7m, 1.8m, 1.9m };
            var gene3Representation = new[] { 2.6m, 2.7m, 2.8m, 2.9m };
            representation1[0] = new Gene<decimal>(gene1Representation, probability);
            representation1[1] = new Gene<decimal>(gene2Representation, probability);
            representation2[0] = new Gene<decimal>(gene1Representation, probability);
            representation2[1] = new Gene<decimal>(gene3Representation, probability);

            var x = new Chromosome<decimal>(representation1);
            var y = new Chromosome<decimal>(representation2);
            var crossover = new Crossover(0.1m);

            var crossed = crossover.RunCrossover(x, y, probability);

            Assert.AreEqual(crossed.Representation[1].Representation[0], gene3Representation[0]);
            Assert.AreEqual(crossed.Representation[1].Representation[1], gene3Representation[1]);
            Assert.AreEqual(crossed.Representation[1].Representation[2], gene3Representation[2]);
            Assert.AreEqual(crossed.Representation[1].Representation[3], gene3Representation[3]);
        }

        [TestMethod]
        public void Crossover_TwoChromosomesOnePercentProbability_CrossoverNotHappend()
        {
            var probability = 0.1m;
            var representation1 = new IGene<decimal>[2];
            var representation2 = new IGene<decimal>[2];

            var gene1Representation = new[] { 1.2m, 1.3m, 1.4m, 1.5m };
            var gene2Representation = new[] { 1.6m, 1.7m, 1.8m, 1.9m };
            var gene3Representation = new[] { 2.6m, 2.7m, 2.8m, 2.9m };
            representation1[0] = new Gene<decimal>(gene1Representation, probability);
            representation1[1] = new Gene<decimal>(gene2Representation, probability);
            representation2[0] = new Gene<decimal>(gene1Representation, probability);
            representation2[1] = new Gene<decimal>(gene3Representation, probability);

            var x = new Chromosome<decimal>(representation1);
            var y = new Chromosome<decimal>(representation2);
            var crossover = new Crossover(0.01m);

            var crossed = crossover.RunCrossover(x, y, probability);

            Assert.AreEqual(crossed.Representation[1].Representation[0], gene2Representation[0]);
            Assert.AreEqual(crossed.Representation[1].Representation[1], gene2Representation[1]);
            Assert.AreEqual(crossed.Representation[1].Representation[2], gene2Representation[2]);
            Assert.AreEqual(crossed.Representation[1].Representation[3], gene2Representation[3]);
        }

    }
}
