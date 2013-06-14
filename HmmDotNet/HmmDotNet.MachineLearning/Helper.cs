using System.Collections.Generic;
using System.Globalization;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning
{
    public static class Helper
    {
        public static IList<IObservation> Convert(double[][] observations)
        {
            var result = new List<IObservation>();
            for (var i = 0; i < observations.Length; i++)
            {
                result.Add(new Observation(observations[i], i.ToString(CultureInfo.InvariantCulture)));
            }
            return result;
        }

        public static IList<IObservation> Convert(double[] observations)
        {
            var result = new List<IObservation>();
            for (var i = 0; i < observations.Length; i++)
            {
                result.Add(new Observation(new [] {observations[i]}, i.ToString(CultureInfo.InvariantCulture)));
            }
            return result;
        }

        public static IList<IState> GetStates<TDistribution>(this IHiddenMarkovModel<TDistribution> model) where TDistribution : IDistribution
        {
            var result = new List<IState>();

            for (int n = 0; n < model.N; n++)
            {
                result.Add(new State(n, n.ToString()));
            }
            return result;
        }

        public static IDistribution[] GetEmissions<TDistribution>(this IHiddenMarkovModel<TDistribution> model) where TDistribution : IDistribution
        {
            var result = new IDistribution[model.N];

            for (int n = 0; n < model.N; n++)
            {
                result[n] = model.Emission[n];
            }
            return result;
        }
    }
}
