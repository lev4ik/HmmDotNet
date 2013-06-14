using System;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;
using HmmDotNet.Statistics.Distributions.Univariate;

namespace HmmDotNet.MachineLearning.HiddenMarkovModels
{
    public static class HiddenMarkovModelFactory
    {
        /// <summary>
        ///     Creates Hidden Markov Model based on it's distribution 
        /// </summary>
        /// <typeparam name="TDistribution"></typeparam>
        /// <param name="parameters"></param>
        /// <returns></returns>
        public static IHiddenMarkovModel<TDistribution> GetModel<TDistribution>(IModelCreationParameters<TDistribution> parameters) where TDistribution : IDistribution
        {
            if (typeof(TDistribution).Name == "Mixture`1")
            {
                switch (typeof(TDistribution).GetGenericArguments()[0].FullName)
                {
                    case "HmmDotNet.Statistics.Distributions.IMultivariateDistribution":
                        return (IHiddenMarkovModel<TDistribution>)new HiddenMarkovModelMixtureDistribution((IModelCreationParameters<Mixture<IMultivariateDistribution>>)parameters);
                }
            }
            else
            {
                switch (typeof(TDistribution).FullName)
                {
                    case "HmmDotNet.Statistics.Distributions.IMultivariateDistribution":
                        return (IHiddenMarkovModel<TDistribution>)new HiddenMarkovModelMultivariateGaussianDistribution((IModelCreationParameters<IMultivariateDistribution>)parameters);
                    case "HmmDotNet.Statistics.Distributions.Univariate.DiscreteDistribution":
                        return (IHiddenMarkovModel<TDistribution>)new HiddenMarkovModel((IModelCreationParameters<DiscreteDistribution>)parameters);
                    case "HmmDotNet.Statistics.Distributions.Univariate.NormalDistribution":
                        return (IHiddenMarkovModel<TDistribution>)new HiddenMarkovModelGaussianDistribution((IModelCreationParameters<Statistics.Distributions.Univariate.NormalDistribution>)parameters);                    
                }                
            }
            throw new InvalidOperationException("Type passed for parameter T is not supported with any implemented Hidden Markov Model");
        }

    }
}
