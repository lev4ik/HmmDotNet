using System;
using System.Collections.Generic;
using HmmDotNet.Logic.Test.MachineLearning.Data;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Algorithms.Search;
using HmmDotNet.MachineLearning.Algorithms.Search.Base;
using HmmDotNet.MachineLearning.Algorithms.Search.Genetic;
using HmmDotNet.MachineLearning.Algorithms.Search.Genetic.PopulationInitialization;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HmmDotNet.MachineLearning.Test.Algorithms.Search
{
    [TestClass]
    public class GeneticSolverTest
    {
        private const decimal MutationProbability = 0.01m;
        private const decimal CrossoverProbability = 0.2m;
        private const int TournamentSize = 10;
        private const int PopulationSize = 100;

        private const int NumberOfComponents = 4;
        private const int NumberOfStates = 4;
        private const int NumberOfIterations = 10;
        private const int LikelihoodTolerance = 20;

        [TestMethod]
        public void GeneticSolver_AllDependencies_SolverCreatedWithAllDependecnies()
        {
            var selectionMethod = new TournamentSelection(TournamentSize);
            var crossoverAlgorithm = new Crossover(CrossoverProbability);
            var mutationAlgorithm = new Mutator(MutationProbability);

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>> { Pi = new double[3], TransitionProbabilityMatrix = new double[3][], Emissions = new Mixture<IMultivariateDistribution>[3]});
            var evaluator = new HmmEvaluator<Mixture<IMultivariateDistribution>>(model, new ForwardBackward(true));
            
            var parameters = new GeneticSolverParameters();

            var solver = new GeneticSolver(parameters, mutationAlgorithm, crossoverAlgorithm, evaluator, selectionMethod);

            Assert.IsNotNull(solver.MutationAlgorithm);
            Assert.IsNotNull(solver.CrossoverAlgorithm);
            Assert.IsNotNull(solver.EvaluationMethod);
            Assert.IsNotNull(solver.SelectionMethod);
        }

        [TestMethod]
        public void Solve_PopulationSizeZero_ZeroSizePopulationReturned()
        {
            var selectionMethod = new TournamentSelection(TournamentSize);
            var crossoverAlgorithm = new Crossover(CrossoverProbability);
            var mutationAlgorithm = new Mutator(MutationProbability);

            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>> { Pi = new double[3], TransitionProbabilityMatrix = new double[3][], Emissions = new Mixture<IMultivariateDistribution>[3] });
            var evaluator = new HmmEvaluator<Mixture<IMultivariateDistribution>>(model, new ForwardBackward(true));

            var parameters = new GeneticSolverParameters();

            var solver = new GeneticSolver(parameters, mutationAlgorithm, crossoverAlgorithm, evaluator, selectionMethod);

            var result = solver.Solve(new List<IChromosome<decimal>>());

            Assert.AreEqual(result.Count, 0);
        }

        [TestMethod]
        public void Solve_InitialPopulationFromTimeSeriesAndTrainedHmmEvaluator_SolvedPopulation()
        {
            var util = new TestDataUtils();
            var series = util.GetSvcData(util.GOOGFilePath, new DateTime(2010, 12, 18), new DateTime(2011, 12, 18));
            var test = util.GetSvcData(util.GOOGFilePath, new DateTime(2011, 12, 18), new DateTime(2012, 01, 18));
            var model = (HiddenMarkovModelMixtureDistribution)HiddenMarkovModelFactory.GetModel(new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfComponents = NumberOfComponents, NumberOfStates = NumberOfStates });
            model.Normalized = true;
            model.Train(series, NumberOfIterations, LikelihoodTolerance);

            var selectionMethod = new TournamentSelection(TournamentSize);
            var crossoverAlgorithm = new Crossover(CrossoverProbability);
            var mutationAlgorithm = new Mutator(MutationProbability);

            var evaluator = new HmmEvaluator<Mixture<IMultivariateDistribution>>(model, new ForwardBackward(true));
            
            var parameters = new GeneticSolverParameters
                {
                    CrossOverProbability = CrossoverProbability,
                    MutationProbability = MutationProbability,
                    NumberOfGenerations = 10,
                    PopulationSize = PopulationSize,
                    TournamentSize = TournamentSize
                };
            var populationInitializer = new FromTimeSeriesRandomInitializer(series);
            var population = populationInitializer.Initialize<decimal>(parameters.PopulationSize, test.Length, MutationProbability);
            for (int i = 0; i < population.Count; i++)
            {
                var chromosome = population[i];
                population[i].FintnessValue = evaluator.Evaluate(chromosome);
            }
            var solver = new GeneticSolver(parameters, mutationAlgorithm, crossoverAlgorithm, evaluator, selectionMethod);

            var result = solver.Solve(population);

            Assert.AreEqual(result.Count, parameters.PopulationSize);            
        }
    }
}
