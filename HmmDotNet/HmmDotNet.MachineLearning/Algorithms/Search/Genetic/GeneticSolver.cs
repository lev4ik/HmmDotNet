using System;
using System.Collections.Generic;
using HmmDotNet.Extentions;
using HmmDotNet.MachineLearning.Algorithms.Search.Base;

namespace HmmDotNet.MachineLearning.Algorithms.Search
{
    public class GeneticSolver : IGeneticSolver
    {
        #region Private Variables

        private IGeneticSolverParameters _parameters;
        private int _numberOfIterations;

        #endregion Private Variables

        #region Public Variables

        public ISelector SelectionMethod { get; protected set; }

        public IEvaluator EvaluationMethod { get; protected set; }

        public ICrossover CrossoverAlgorithm { get; protected set; }

        public IMutatable MutationAlgorithm { get; protected set; }
        
        #endregion Public Variables

        public GeneticSolver(IGeneticSolverParameters parameters, IMutatable mutator, ICrossover crossover, IEvaluator evaluator, ISelector selector)
        {
            _numberOfIterations = 0;
            _parameters = parameters;
            EvaluationMethod = evaluator;
            CrossoverAlgorithm = crossover;
            SelectionMethod = selector;
            MutationAlgorithm = mutator;
        }

        public IList<IChromosome<T>> Solve<T>(IList<IChromosome<T>> population) 
        {
            if (population.Count == 0)
            {
                return population;
            }

            var currentPopulation = population.Clone();

            do
            {
                var newPopulation = new List<IChromosome<T>>();
                while (newPopulation.Count < _parameters.PopulationSize)
                {
                    var selected = SelectionMethod.Selection(currentPopulation);
                    var chromozome = CrossoverAlgorithm.RunCrossover(selected.Item1, selected.Item2, _parameters.CrossOverProbability);
                    chromozome = MutationAlgorithm.RunMutation(chromozome, _parameters.MutationProbability);
                    chromozome.FintnessValue = EvaluationMethod.Evaluate(chromozome);
                    newPopulation.Add(chromozome);
                }
                currentPopulation = newPopulation;

            } while (!Convirged());

            return currentPopulation;
        }

        public bool Convirged()
        {
            return (_numberOfIterations <= _parameters.NumberOfGenerations);
        }
    }
}
