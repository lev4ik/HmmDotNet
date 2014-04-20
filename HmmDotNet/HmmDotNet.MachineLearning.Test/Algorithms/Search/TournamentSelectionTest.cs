using System.Collections.Generic;
using HmmDotNet.MachineLearning.Algorithms.Search;
using HmmDotNet.MachineLearning.Algorithms.Search.Base;
using HmmDotNet.MachineLearning.Algorithms.Search.Genetic;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HmmDotNet.MachineLearning.Test.Algorithms.Search
{
    [TestClass]
    public class TournamentSelectionTest
    {
        [TestMethod]
        public void Selection_TournamentSizeSameAsPopulation_TwoBestFittedChromosomes()
        {
            var population = new List<IChromosome<decimal>>();
            var chromosome1 = new Chromosome<decimal>(new IGene<decimal>[] {new Gene<decimal>(new []{1.2m, 1.3m}, 0.1m), 
                                                                            new Gene<decimal>(new []{1.3m, 1.5m}, 0.1m)});
            chromosome1.FintnessValue = 90;
            var chromosome2 = new Chromosome<decimal>(new IGene<decimal>[] {new Gene<decimal>(new []{2.2m, 2.3m}, 0.1m), 
                                                                            new Gene<decimal>(new []{2.3m, 2.5m}, 0.1m)});
            chromosome2.FintnessValue = 80;
            var chromosome3 = new Chromosome<decimal>(new IGene<decimal>[] {new Gene<decimal>(new []{3.2m, 3.3m}, 0.1m), 
                                                                            new Gene<decimal>(new []{3.3m, 3.5m}, 0.1m)});
            chromosome3.FintnessValue = 70;

            population.Add(chromosome1);
            population.Add(chromosome2);
            population.Add(chromosome3);
            var selector = new TournamentSelection(3);

            var result = selector.Selection(population);

            Assert.IsTrue(result.Item1.FintnessValue == 90 || result.Item2.FintnessValue == 90);
        }

        [TestMethod]
        public void Selection_TournamentSize5AndPopulationSize10_BestFittedChromosomes()
        {
            var population = new List<IChromosome<decimal>>();
            for (int i = 0; i < 10; i++)
            {
                var chromosome = new Chromosome<decimal>(new IGene<decimal>[] {new Gene<decimal>(new []{i + .2m, i + .3m}, 0.1m), 
                                                                               new Gene<decimal>(new []{i + .3m, i + .5m}, 0.1m)});
                chromosome.FintnessValue = 5 * i;
                population.Add(chromosome);
            }

            var selector = new TournamentSelection(5);

            var selected = selector.Selection(population);

            Assert.IsTrue(selected.Item1.FintnessValue > 30 || selected.Item2.FintnessValue > 30);
        }
    }
}
