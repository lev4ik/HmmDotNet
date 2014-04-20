using System;
using System.Collections.Generic;
using HmmDotNet.MachineLearning.Algorithms.Search.Base;

namespace HmmDotNet.MachineLearning.Algorithms.Search.Genetic.PopulationInitialization
{
    public class FromTimeSeriesRandomInitializer : IPopulationInitializer
    {
        private double[][] _trainigSet;

        public FromTimeSeriesRandomInitializer(double[][] trainigSet)
        {
            _trainigSet = trainigSet;
        }

        public IList<IChromosome<T>> Initialize<T>(int populationSize, int chromosomeSize, decimal mutationProbability)
        {
            var rd = new Random();
            var population = new List<IChromosome<T>>();
            for (var i = 0; i < populationSize; i++)
            {
                var represenatation = new IGene<T>[chromosomeSize];
                for (int j = 0; j < chromosomeSize; j++)
                {
                    var index = rd.Next(0, _trainigSet.Length - 1);
                    var gene = Array.ConvertAll(_trainigSet[index], x => (T)Convert.ChangeType(x, typeof(T)));

                    represenatation[j] = new Gene<T>(gene, mutationProbability);
                }
                var chromosome = new Chromosome<T>(represenatation);
                population.Add(chromosome);
            }
            return population;
        }
    }
}
