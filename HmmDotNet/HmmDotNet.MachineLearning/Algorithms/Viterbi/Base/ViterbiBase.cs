using System;
using System.Collections.Generic;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Univariate;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public abstract class ViterbiBase : IDynamicProgrammingAlgorithm
    {
        #region Properties

        public double[][] ComputationPath { get; protected set; }
        public int[][] ReproducePath { get; protected set; }

        protected bool Normalized { get; set; }

        #endregion Properties

        public void PrintComputationalPath()
        {
            if (ComputationPath != null)
            {
                var T = ComputationPath.Length;
                var N = ComputationPath[0].Length;

                for (var t = 0; t < T; t++)
                {
                    for (var i = 0; i < N; i++)
                    {
                        Console.Write(string.Format("[{0}|{1:0.00000000000}]", i, ComputationPath[t][i]));
                    }
                    Console.WriteLine();
                }
            }
        }

        protected static int GetMaximumDeltaValueStateIndex(double[] vector)
        {
            var max = double.NegativeInfinity;
            var maxIndex = 0;
            for (var i = 0; i < vector.Length; i++)
            {
                if (max <= vector[i])
                {
                    max = vector[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }

        protected static int GetMinimumIndex(ViterbiDataNode[] vector)
        {
            var min = 0d;
            var minIndex = 0;
            for (var i = 0; i < vector.Length; i++)
            {
                if (min >= vector[i].Value)
                {
                    min = vector[i].Value;
                    minIndex = i;
                }
            }
            return minIndex;
        }

        #region Private Methods

        protected static double GetProbability(IDistribution distribution, IList<IObservation> observations, int place)
        {
            var d = distribution as DiscreteDistribution;
            if (d == null)
            {
                return distribution.ProbabilityDensityFunction(observations[place].Value);
            }
            return distribution.ProbabilityMassFunction(observations[place].Value);
        }

        #endregion Private Methods
    }
}
