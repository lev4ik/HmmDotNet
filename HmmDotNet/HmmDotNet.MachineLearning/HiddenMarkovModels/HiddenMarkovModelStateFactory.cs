﻿using System;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.Estimators;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;
using HmmDotNet.Statistics.Distributions.Univariate;

namespace HmmDotNet.MachineLearning.HiddenMarkovModels
{
    public static class HiddenMarkovModelStateFactory
    {
        /// <summary>
        /// 
        /// </summary>
        /// <typeparam name="TDistribution"></typeparam>
        /// <param name="parameters"></param>
        /// <returns></returns>
        public static IHiddenMarkovModel<TDistribution> GetState<TDistribution>(IModelCreationParameters<TDistribution> parameters) where TDistribution : IDistribution
        {
            if (typeof(TDistribution).Name == "Mixture`1")
            {
                switch (typeof (TDistribution).GetGenericArguments()[0].FullName)
                {
                    case "HmmDotNet.Statistics.Distributions.IMultivariateDistribution":
                        return (IHiddenMarkovModel<TDistribution>)new HiddenMarkovModel<Mixture<IMultivariateDistribution>>((IModelCreationParameters<Mixture<IMultivariateDistribution>>)parameters);
                    case "HmmDotNet.Statistics.Distributions.Multivariate.NormalDistribution":
                        return (IHiddenMarkovModel<TDistribution>)new HiddenMarkovModel<Mixture<Statistics.Distributions.Multivariate.NormalDistribution>>((IModelCreationParameters<Mixture<Statistics.Distributions.Multivariate.NormalDistribution>>)parameters);
                }
            }
            else
            {
                switch (typeof(TDistribution).FullName)
                {
                    case "HmmDotNet.Statistics.Distributions.IMultivariateDistribution":
                        return (IHiddenMarkovModel<TDistribution>)new HiddenMarkovModel<IMultivariateDistribution>((IModelCreationParameters<IMultivariateDistribution>)parameters);
                    case "HmmDotNet.Statistics.Distributions.Univariate.DiscreteDistribution":
                        return (IHiddenMarkovModel<TDistribution>)new HiddenMarkovModel<DiscreteDistribution>((IModelCreationParameters<DiscreteDistribution>)parameters);
                    case "HmmDotNet.Statistics.Distributions.Multivariate.NormalDistribution":
                        return (IHiddenMarkovModel<TDistribution>)new HiddenMarkovModel<Statistics.Distributions.Multivariate.NormalDistribution>((IModelCreationParameters<Statistics.Distributions.Multivariate.NormalDistribution>)parameters);
                    case "HmmDotNet.Statistics.Distributions.Univariate.NormalDistribution":
                        return (IHiddenMarkovModel<TDistribution>)new HiddenMarkovModel<Statistics.Distributions.Univariate.NormalDistribution>((IModelCreationParameters<Statistics.Distributions.Univariate.NormalDistribution>)parameters);
                }
            }
            throw new InvalidOperationException("Type passed for parameter T is not supported with any implemented Hidden Markov Model");
        }

    }
}
