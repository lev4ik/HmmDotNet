using HmmDotNet.Mathematic.Extentions;

namespace HmmDotNet.Statistics.Distributions
{
    public abstract class MultivariateDiscreteDistribution : IMultivariateDistribution
    {

        #region IMultivariateDistribution Members

        public abstract double[] Variance { get; set; }

        public abstract double[] Mean { get; set; }

        public abstract double[,] Covariance { get; set; }

        public abstract int Dimension { get; set; }

        #endregion

        #region IDistribution Members

        public abstract double ProbabilityMassFunction(params double[] x);

        public double ProbabilityDensityFunction(params double[] x)
        {
            throw new System.NotImplementedException();
        }

        public abstract IDistribution Evaluate(double[] observations);

        public abstract IDistribution Evaluate(double[][] observations, out double likelihood);

        public abstract IDistribution Evaluate(double[] observations, double[] weights);

        public abstract IDistribution Evaluate(double[][] observations, double[] weights, out double likelihood);

        #endregion

        #region ICloneable Members

        public abstract object Clone();

        #endregion

        public bool Equals(IDistribution other)
        {
            var d = other as MultivariateDiscreteDistribution;
            if (d == null)
            {
                //throw new ApplicationException("Incompatable types, other is not UnivariateDiscreteDistribution");
                return false;
            }

            return Variance.EqualsTo(d.Variance) &&
                   Mean.EqualsTo(d.Mean) && 
                   Covariance.EqualsTo(d.Covariance) && 
                   Dimension == d.Dimension;
        }
    }
}
