using System;
using HmmDotNet.Mathematic.Extentions;

namespace HmmDotNet.Statistics.Distributions
{
    public abstract class UnivariateContinuousDistribution : IUnivariateDistribution
    {

        #region IUnivariateDistribution Members

        public abstract double Variance { get;}

        public abstract double Mean { get; }

        #endregion

        #region IDistribution Members

        public double ProbabilityMassFunction(params double[] x)
        {
            throw new System.NotImplementedException();
        }

        public abstract double ProbabilityDensityFunction(params double[] x);

        public abstract IDistribution Evaluate(double[] observations);

        public abstract IDistribution Evaluate(double[][] observations, out double likelihood);

        public abstract IDistribution Evaluate(double[] observations, double[] weights);

        public abstract IDistribution Evaluate(double[][] observations, double[] weights, out double likelihood);

        #endregion

        #region ICloneable Members

        public abstract object Clone();

        public bool Equals(IDistribution other)
        {
            var d = other as UnivariateContinuousDistribution;
            if (d == null)
            {
                //throw new ApplicationException("Incompatable types, other is not UnivariateDiscreteDistribution");
                return false;
            }

            return !(Variance.EqualsTo(d.Variance) || Mean.EqualsTo(d.Mean));
        }
        #endregion
    }
}
