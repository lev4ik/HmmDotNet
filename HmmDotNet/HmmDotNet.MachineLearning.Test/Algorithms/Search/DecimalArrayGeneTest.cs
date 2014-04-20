using HmmDotNet.MachineLearning.Algorithms.Search;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HmmDotNet.MachineLearning.Test.Algorithms.Search
{
    [TestClass]
    public class DecimalArrayGeneTest
    {
        [TestMethod]
        public void Create_Gene_NewGene()
        {
            var representation = new [] { 1.2m, 1.3m, 1.4m, 1.5m };
            var probability = 0.1m;
            var gene = new Gene<decimal>(representation, probability);

            Assert.AreEqual(gene.Representation[0], representation[0]);
            Assert.AreEqual(gene.Representation[1], representation[1]);
            Assert.AreEqual(gene.Representation[2], representation[2]);
            Assert.AreEqual(gene.Representation[3], representation[3]);

            Assert.AreEqual(gene.MutationProbability, probability);
        }

        [TestMethod]
        public void Clone_Gene_ClonedGene()
        {
            var representation = new [] { 1.2m, 1.3m, 1.4m, 1.5m};
            var probability = 0.1m;
            var gene = new Gene<decimal>(representation, probability);

            var cloned = (Gene<decimal>)gene.Clone();

            Assert.AreEqual(cloned.Representation[0], representation[0]);
            Assert.AreEqual(cloned.Representation[1], representation[1]);
            Assert.AreEqual(cloned.Representation[2], representation[2]);
            Assert.AreEqual(cloned.Representation[3], representation[3]);

            Assert.AreEqual(cloned.MutationProbability, probability);
        }

        [TestMethod]
        public void Mutate_GeneAndOnePercentProbability_NoMutation()
        {
            var representation = new[] { 1.2m, 1.3m, 1.4m, 1.5m };
            var probability = 0.1m;
            var gene = new Gene<decimal>(representation, 0.01m);

            gene.Mutate(probability);
            var exp = gene.Representation[0] == representation[0] || gene.Representation[1] == representation[1] || gene.Representation[2] == representation[2] || gene.Representation[3] == representation[3];
            
            Assert.IsTrue(exp);
        }

        [TestMethod]
        public void Mutate_GeneAndOnePercentProbability_Mutated()
        {
            var representation = new[] { 1.2m, 1.3m, 1.4m, 1.5m };
            var probability = 0.1m;
            var gene = new Gene<decimal>(representation, 0.01m);

            gene.Mutate(probability);

            Assert.AreEqual(gene.Representation[0], representation[0]);
            Assert.AreEqual(gene.Representation[1], representation[1]);
            Assert.AreEqual(gene.Representation[2], representation[2]);
            Assert.AreEqual(gene.Representation[3], representation[3]);
        }
    }
}
