using System;
using System.Collections.Generic;
using System.Linq;
using HmmDotNet.MachineLearning.Algorithms.Search.Base;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms.Search.Genetic
{
    public class HmmEvaluator<TEmmisionType> : IEvaluator where TEmmisionType : IDistribution
    {
        private readonly IForwardBackward _fitnessClaculator;
        private readonly IHiddenMarkovModel<TEmmisionType> _model;

        public HmmEvaluator(IHiddenMarkovModel<TEmmisionType> model, IForwardBackward fitnessClaculator)
        {
            _fitnessClaculator = fitnessClaculator;
            _model = model;
        }

        public decimal Evaluate<T>(IChromosome<T> c)
        {
            var observations = GetObservationSequence(c);
            var fitnessValue = _fitnessClaculator.RunForward(observations, _model);
            return (decimal)fitnessValue;
        }

        private IList<IObservation> GetObservationSequence<T>(IChromosome<T> c)
        {
            return (from o in c.Representation
                    select new Observation(Array.ConvertAll(o.Representation, x => (double)Convert.ChangeType(x, typeof(double))), "")).ToList<IObservation>();
        }
    }
}
