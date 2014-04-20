using HmmDotNet.MachineLearning.Algorithms.Search;
using HmmDotNet.MachineLearning.Algorithms.Search.Base;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HmmDotNet.MachineLearning.Test.Algorithms.Search
{
    [TestClass]
    public class DecimalArrayChromosomeTest
    {
        [TestMethod]
        public void Create_Chromosome_NewChromosome()
        {
            var probability = 0.1m;
            var representation = new IGene<decimal>[3];
            var gene1Representation = new[] { 1.2m, 1.3m, 1.4m, 1.5m };
            var gene2Representation = new[] { 1.6m, 1.7m, 1.8m, 1.9m };
            representation[0] = new Gene<decimal>(gene1Representation, probability);
            representation[1] = new Gene<decimal>(gene2Representation, probability);

            var chromosome = new Chromosome<decimal>(representation);

            Assert.AreEqual(chromosome.Representation[0].Representation[0], gene1Representation[0]);
            Assert.AreEqual(chromosome.Representation[0].Representation[1], gene1Representation[1]);
            Assert.AreEqual(chromosome.Representation[0].Representation[2], gene1Representation[2]);
            Assert.AreEqual(chromosome.Representation[0].Representation[3], gene1Representation[3]);

            Assert.AreEqual(chromosome.Representation[1].Representation[0], gene2Representation[0]);
            Assert.AreEqual(chromosome.Representation[1].Representation[1], gene2Representation[1]);
            Assert.AreEqual(chromosome.Representation[1].Representation[2], gene2Representation[2]);
            Assert.AreEqual(chromosome.Representation[1].Representation[3], gene2Representation[3]);
        }

        [TestMethod]
        public void Clone_Chromosome_ClonedChromosome()
        {
            var probability = 0.1m;
            var representation = new IGene<decimal>[3];
            var gene1Representation = new[] { 1.2m, 1.3m, 1.4m, 1.5m };
            var gene2Representation = new[] { 1.6m, 1.7m, 1.8m, 1.9m };
            representation[0] = new Gene<decimal>(gene1Representation, probability);
            representation[1] = new Gene<decimal>(gene2Representation, probability);

            var chromosome = new Chromosome<decimal>(representation);
            var cloned = (Chromosome<decimal>)chromosome.Clone();

            Assert.AreEqual(cloned.Representation[0].Representation[0], gene1Representation[0]);
            Assert.AreEqual(cloned.Representation[0].Representation[1], gene1Representation[1]);
            Assert.AreEqual(cloned.Representation[0].Representation[2], gene1Representation[2]);
            Assert.AreEqual(cloned.Representation[0].Representation[3], gene1Representation[3]);

            Assert.AreEqual(cloned.Representation[1].Representation[0], gene2Representation[0]);
            Assert.AreEqual(cloned.Representation[1].Representation[1], gene2Representation[1]);
            Assert.AreEqual(cloned.Representation[1].Representation[2], gene2Representation[2]);
            Assert.AreEqual(cloned.Representation[1].Representation[3], gene2Representation[3]);
        }
    }
}
