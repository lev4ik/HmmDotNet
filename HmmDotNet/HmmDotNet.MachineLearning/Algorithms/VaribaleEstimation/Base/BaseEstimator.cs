using System.Collections.Generic;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public abstract class BaseEstimator
    {
        public bool LogNormalized { get; set; }

        protected static double GetProbability(IDistribution distribution, IList<IObservation> observations, int place)
        {
            if (distribution is UnivariateDiscreteDistribution || distribution is MultivariateDiscreteDistribution)
            {
                return distribution.ProbabilityMassFunction(observations[place].Value);
            }
            return distribution.ProbabilityDensityFunction(observations[place].Value);            
        }
    }
}
