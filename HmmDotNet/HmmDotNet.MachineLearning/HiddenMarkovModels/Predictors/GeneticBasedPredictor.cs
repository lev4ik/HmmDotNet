using System;
using System.Linq;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Algorithms.Search;
using HmmDotNet.MachineLearning.Algorithms.Search.Base;
using HmmDotNet.MachineLearning.Algorithms.Search.Genetic;
using HmmDotNet.MachineLearning.Algorithms.Search.Genetic.PopulationInitialization;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.HiddenMarkovModels.Predictors
{
    public class GeneticBasedPredictor : IHiddenMarkovModelPredictor
    {
        public decimal MutationProbability = 0.01m;
        public decimal CrossoverProbability = 0.2m;
        public int TournamentSize = 10;
        //public int PopulationSize = 100;
        public int NumberOfGenerations = 100;

        public IPredictionResult Predict<TDistribution>(IHiddenMarkovModel<TDistribution> model, IPredictionRequest request) where TDistribution : IDistribution
        {
            var selectionMethod = new TournamentSelection(TournamentSize);
            var crossoverAlgorithm = new Crossover(CrossoverProbability);
            var mutationAlgorithm = new Mutator(MutationProbability);
            var evaluator = new HmmEvaluator<TDistribution>(model, new ForwardBackward(true));
          
            var parameters = new GeneticSolverParameters
            {
                CrossOverProbability = CrossoverProbability,
                MutationProbability = MutationProbability,
                NumberOfGenerations = NumberOfGenerations,
                PopulationSize = request.NumberOfDays * 10,
                TournamentSize = TournamentSize
            };
            var predictions = new PredictionResult { Predicted = new double[request.NumberOfDays][] };

            var solver = new GeneticSolver(parameters, mutationAlgorithm, crossoverAlgorithm, evaluator, selectionMethod);
            var populationInitializer = new FromTimeSeriesRandomInitializer(request.TrainingSet);
            var population = populationInitializer.Initialize<decimal>(parameters.PopulationSize, request.NumberOfDays, MutationProbability);
            for (int i = 0; i < population.Count; i++)
            {
                var chromosome = population[i];
                population[i].FintnessValue = evaluator.Evaluate(chromosome);
            }

            var result = solver.Solve(population);
            // Get best fitted chromosome
            var maximumFitness = result[0].FintnessValue;
            var solution = result[0];

            foreach (var chromosome in result)
            {
                if (maximumFitness <= chromosome.FintnessValue)
                {
                    solution = chromosome;
                    maximumFitness = chromosome.FintnessValue;
                }
            }
            // Convert it to array
            for (int i = 0; i < solution.Representation.Length; i++)
            {
                predictions.Predicted[i] = Array.ConvertAll(solution.Representation[i].Representation, x => (double)Convert.ChangeType(x, typeof(double)));
            }

            return predictions;
        }

        public IEvaluationResult Evaluate(IEvaluationRequest request)
        {
            throw new NotImplementedException();
        }
    }
}
