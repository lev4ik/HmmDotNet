using System.Collections.Generic;
using System.Linq;
using HmmDotNet.Extentions.Data;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms.Search
{
    public class BeamLikeSearch
    {
        public double[][] Search<TDistribution>(double[][] population, int k, int n, int numberOfIterations, MaximumChangeRatios ratios, IHiddenMarkovModel<TDistribution> model) 
            where TDistribution : IDistribution
        {
            var current = ExpandCurrentPopulation(population, n, ratios.Down, ratios.Up); ;            

            for (var i = 0; i < numberOfIterations; i++)
            {
                var next = GenerateNextStagePopulation(current, k, n, ratios, model);
                current = next;
            }

            return current; 
        }

        public double[][] GenerateNextStagePopulation<TDistribution>(double[][] current, int k, int n, MaximumChangeRatios ratios, IHiddenMarkovModel<TDistribution> model)
            where TDistribution : IDistribution
        {
            var dic = new SortedList<double, double[]>();

            for (var j = 0; j < current.Length; j++)
            {
                var h = HeuristicFunction(current[j], model);
                if (dic.Count < k)
                {
                    dic.Add(h, current[j]);
                }
                else
                {
                    if (dic.Keys[dic.Count] < h)
                    {
                        dic.Remove(dic.Keys[dic.Count]);
                        dic.Add(h, current[j]);
                    }
                }
            }

            current = (from e in dic select e.Value).ToArray();
            current = ExpandCurrentPopulation(current, n, ratios.Down, ratios.Up);
            return current;
        }

        public double HeuristicFunction<TDistribution>(double[] node, IHiddenMarkovModel<TDistribution> model)
            where TDistribution : IDistribution
        {
            //var arr = trainingSet.Concat(new []{ node });
            var forwardBackward = new ForwardBackward(model.Normalized);
            var h = forwardBackward.RunForward(Helper.Convert(new[] { node }), model);
            return h;
        }

        public double[][] ExpandCurrentPopulation(double[][] population, int populationLength, double minPercent, double maxPercent)
        {
            var generation = new List<double[]>();

            for (var i = 0; i < population.Length; i++)
            {
                var intervalOpen = (((population[i][0] + population[i][0] * maxPercent) - (population[i][0] - population[i][0] * minPercent)) / populationLength) / 100;
                var intervalHigh = (((population[i][1] + population[i][1] * maxPercent) - (population[i][1] - population[i][1] * minPercent)) / populationLength) / 100;
                var intervalLow = (((population[i][2] + population[i][2] * maxPercent) - (population[i][2] - population[i][2] * minPercent)) / populationLength) / 100;
                var intervalClose = (((population[i][3] + population[i][3] * maxPercent) - (population[i][3] - population[i][3] * minPercent)) / populationLength) / 100;

                var openStartValue = population[i][0] - (population[i][0] * minPercent) / 100;
                var highStartValue = population[i][1] - (population[i][1] * minPercent) / 100;
                var lowStartValue = population[i][2] - (population[i][2] * minPercent) / 100;
                var closeStartValue = population[i][3] - (population[i][3] * minPercent) / 100;

                for (var j = 0; j < populationLength; j++)
                {
                    generation.Add(new double[]
                        {
                            openStartValue + j * intervalOpen,
                            highStartValue + j * intervalHigh,
                            lowStartValue + j * intervalLow,
                            closeStartValue + j * intervalClose
                        });
                }
            }

            return generation.ToArray();
        }
    }
}
