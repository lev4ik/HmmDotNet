using System;
using System.Collections.Generic;
using HmmDotNet.Extentions;
using HmmDotNet.MachineLearning.Algorithms.Search.Base;

namespace HmmDotNet.MachineLearning.Algorithms.Search.Genetic
{
    public class TournamentSelection : ISelector
    {
        private readonly int _tournamentSize;
        private Random _rd = new Random();

        public TournamentSelection(int tournamentSize)
        {
            _tournamentSize = tournamentSize;         
        }

        public Tuple<IChromosome<T>, IChromosome<T>> Selection<T>(IList<IChromosome<T>> population)
        {
            var matingPool = CreateMatingPool(population);

            while (matingPool.Count != 2)
            {
                var index1 = _rd.Next(0, matingPool.Count);
                var index2 = _rd.Next(0, matingPool.Count);

                if (index1 == index2)
                {
                    if (index2 == 0)
                        index2++;
                    else
                        index2--;
                }
                matingPool.Remove(matingPool[index1].FintnessValue <= matingPool[index2].FintnessValue
                                      ? matingPool[index1]
                                      : matingPool[index2]);
            }

            return new Tuple<IChromosome<T>, IChromosome<T>>(matingPool[0], matingPool[1]);
        }

        private IList<IChromosome<T>> CreateMatingPool<T>(IList<IChromosome<T>> population)
        {
            var populationCopy = population.Copy();
            var matingPool = new List<IChromosome<T>>(_tournamentSize);

            for (var i = 0; i < _tournamentSize; i++)
            {
                var index = _rd.Next(0, population.Count - i);
                matingPool.Add(populationCopy[index]);
                populationCopy.RemoveAt(index);
            }
            return matingPool;
        }
    }
}
