using System;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.Statistics.Examples
{
    public class RainyDistribution : IDistribution
    {
        #region Properties

        public double Variance
        {
            get { throw new NotImplementedException(); }
        }

        public double Mean
        {
            get { throw new NotImplementedException(); }
        }

        public int Median
        {
            get { throw new NotImplementedException(); }
        }

        public double Skewness
        {
            get { throw new NotImplementedException(); }
        }

        #endregion Properties

        #region Methods

        public double ProbabilityMassFunction(params double[] x)
        {
            throw new NotImplementedException();
        }

        public double ProbabilityDensityFunction(params double[] x)
        {
            double probability = 0;
            if (x[0] == 0)
            {
                probability = 0.1;
            }
            else if (x[0] == 1)
            {
                probability = 0.4;
            }
            else if (x[0] == 2)
            {
                probability = 0.5;
            }
            return probability;
        }

        public IDistribution Evaluate(double[] observations)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[] observations, double[] weights)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, out double likelihood)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, double[] weights, out double likelihood)
        {
            throw new NotImplementedException();
        }

        #endregion Methods

        #region ICloneable Members

        public object Clone()
        {
            throw new NotImplementedException();
        }

        #endregion

        public bool Equals(IDistribution other)
        {
            var d = other as RainyDistribution;
            if (d == null)
            {
                return false;
            }
            return (Mean - d.Mean == 0 && Variance - d.Variance == 0);
        }
    }

    public class SunnyDistribution : IDistribution
    {
        #region Properties

        public double Variance
        {
            get { throw new NotImplementedException(); }
        }

        public double Mean
        {
            get { throw new NotImplementedException(); }
        }

        public int Median
        {
            get { throw new NotImplementedException(); }
        }

        public double Skewness
        {
            get { throw new NotImplementedException(); }
        }

        #endregion Properties

        #region Methods

        public double ProbabilityMassFunction(params double[] x)
        {
            throw new NotImplementedException();
        }

        public double ProbabilityDensityFunction(params double[] x)
        {
            double probability = 0;
            if (x[0] == 0)
            {
                probability = 0.6;
            }
            else if (x[0] == 1)
            {
                probability = 0.3;
            }
            else if (x[0] == 2)
            {
                probability = 0.1;
            }
            return probability;
        }

        public IDistribution Evaluate(double[] observations)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[] observations, double[] weights)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, out double likelihood)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, double[] weights, out double likelihood)
        {
            throw new NotImplementedException();
        }

        #endregion Methods

        #region ICloneable Members

        public object Clone()
        {
            throw new NotImplementedException();
        }

        #endregion

        public bool Equals(IDistribution other)
        {
            var d = other as SunnyDistribution;
            if (d == null)
            {
                return false;
            }
            return (Mean - d.Mean == 0 && Variance - d.Variance == 0);
        }
    }

    public class FirstDistribution : IDistribution
    {
        #region Properties

        public double Variance
        {
            get { throw new NotImplementedException(); }
        }

        public double Mean
        {
            get { throw new NotImplementedException(); }
        }

        public int Median
        {
            get { throw new NotImplementedException(); }
        }

        public double Skewness
        {
            get { throw new NotImplementedException(); }
        }

        #endregion Properties

        #region Methods

        public double ProbabilityMassFunction(params double[] x)
        {
            throw new NotImplementedException();
        }

        public double ProbabilityDensityFunction(params double[] x)
        {
            double probability = 0;
            if (x[0] == 0)
            {
                probability = 0.5;
            }
            else if (x[0] == 1)
            {
                probability = 0.5;
            }
            return probability;
        }

        public IDistribution Evaluate(double[] observations)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[] observations, double[] weights)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, out double likelihood)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, double[] weights, out double likelihood)
        {
            throw new NotImplementedException();
        }

        #endregion Methods

        #region ICloneable Members

        public object Clone()
        {
            throw new NotImplementedException();
        }

        #endregion

        public bool Equals(IDistribution other)
        {
            var d = other as FirstDistribution;
            if (d == null)
            {
                return false;
            }
            return (Mean - d.Mean == 0 && Variance - d.Variance == 0);
        }
    }

    public class SecondDistribution : IDistribution
    {
        #region Properties

        public double Variance
        {
            get { throw new NotImplementedException(); }
        }

        public double Mean
        {
            get { throw new NotImplementedException(); }
        }

        public int Median
        {
            get { throw new NotImplementedException(); }
        }

        public double Skewness
        {
            get { throw new NotImplementedException(); }
        }

        #endregion Properties

        #region Methods

        public double ProbabilityMassFunction(params double[] x)
        {
            throw new NotImplementedException();
        }

        public double ProbabilityDensityFunction(params double[] x)
        {
            double probability = 0;
            if (x[0] == 0)
            {
                probability = 1 / 3d;
            }
            else if (x[0] == 1)
            {
                probability = 2 / 3d;
            }
            return probability;
        }

        public IDistribution Evaluate(double[] observations)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[] observations, double[] weights)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, out double likelihood)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, double[] weights, out double likelihood)
        {
            throw new NotImplementedException();
        }

        #endregion Methods

        #region ICloneable Members

        public object Clone()
        {
            throw new NotImplementedException();
        }

        #endregion

        public bool Equals(IDistribution other)
        {
            var d = other as SecondDistribution;
            if (d == null)
            {
                return false;
            }
            return (Mean - d.Mean == 0 && Variance - d.Variance == 0);
        }
    }

    public class ThirdDistribution : IDistribution
    {
        #region Properties

        public double Variance
        {
            get { throw new NotImplementedException(); }
        }

        public double Mean
        {
            get { throw new NotImplementedException(); }
        }

        public int Median
        {
            get { throw new NotImplementedException(); }
        }

        public double Skewness
        {
            get { throw new NotImplementedException(); }
        }

        #endregion Properties

        #region Methods

        public double ProbabilityMassFunction(params double[] x)
        {
            throw new NotImplementedException();
        }

        public double ProbabilityDensityFunction(params double[] x)
        {
            double probability = 0;
            if (x[0] == 0)
            {
                probability = 3 / 4d;
            }
            else if (x[0] == 1)
            {
                probability = 1 / 4d;
            }
            return probability;
        }

        public IDistribution Evaluate(double[] observations)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[] observations, double[] weights)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, out double likelihood)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, double[] weights, out double likelihood)
        {
            throw new NotImplementedException();
        }

        #endregion Methods

        #region ICloneable Members

        public object Clone()
        {
            throw new NotImplementedException();
        }

        #endregion

        public bool Equals(IDistribution other)
        {
            var d = other as ThirdDistribution;
            if (d == null)
            {
                return false;
            }
            return (Mean - d.Mean == 0 && Variance - d.Variance == 0);
        }
    }

    public class HDistribution : IDistribution
    {
        #region Properties

        public double Variance
        {
            get { throw new NotImplementedException(); }
        }

        public double Mean
        {
            get { throw new NotImplementedException(); }
        }

        public int Median
        {
            get { throw new NotImplementedException(); }
        }

        public double Skewness
        {
            get { throw new NotImplementedException(); }
        }

        #endregion Properties

        #region Methods

        public double ProbabilityMassFunction(params double[] x)
        {
            throw new NotImplementedException();
        }

        public double ProbabilityDensityFunction(params double[] x)
        {
            double probability = 0;
            if (x[0] == 0)
            {
                probability = 0.2;
            }
            else if (x[0] == 1)
            {
                probability = 0.3;
            }
            else if (x[0] == 2)
            {
                probability = 0.3;
            }
            else if (x[0] == 3)
            {
                probability = 0.2;
            }
            return probability;
        }

        public IDistribution Evaluate(double[] observations)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[] observations, double[] weights)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, out double likelihood)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, double[] weights, out double likelihood)
        {
            throw new NotImplementedException();
        }

        #endregion Methods

        #region ICloneable Members

        public object Clone()
        {
            throw new NotImplementedException();
        }

        #endregion

        public bool Equals(IDistribution other)
        {
            var d = other as HDistribution;
            if (d == null)
            {
                return false;
            }
            return (Mean - d.Mean == 0 && Variance - d.Variance == 0);
        }
    }

    public class LDistribution : IDistribution
    {
        #region Properties

        public double Variance
        {
            get { throw new NotImplementedException(); }
        }

        public double Mean
        {
            get { throw new NotImplementedException(); }
        }

        public int Median
        {
            get { throw new NotImplementedException(); }
        }

        public double Skewness
        {
            get { throw new NotImplementedException(); }
        }

        #endregion Properties

        #region Methods

        public double ProbabilityMassFunction(params double[] x)
        {
            throw new NotImplementedException();
        }

        public double ProbabilityDensityFunction(params double[] x)
        {
            double probability = 0;
            if (x[0] == 0)
            {
                probability = 0.3;
            }
            else if (x[0] == 1)
            {
                probability = 0.2;
            }
            else if (x[0] == 2)
            {
                probability = 0.2;
            }
            else if (x[0] == 3)
            {
                probability = 0.3;
            }
            return probability;
        }

        public IDistribution Evaluate(double[] observations)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[] observations, double[] weights)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, out double likelihood)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, double[] weights, out double likelihood)
        {
            throw new NotImplementedException();
        }

        #endregion Properties

        #region ICloneable Members

        public object Clone()
        {
            throw new NotImplementedException();
        }

        #endregion

        public bool Equals(IDistribution other)
        {
            var d = other as LDistribution;
            if (d == null)
            {
                return false;
            }
            return (Mean - d.Mean == 0 && Variance - d.Variance == 0);
        }
    }

    public class InStateSDistribution : IDistribution
    {
        #region Properties

        public double Variance
        {
            get { throw new NotImplementedException(); }
        }

        public double Mean
        {
            get { throw new NotImplementedException(); }
        }

        public int Median
        {
            get { throw new NotImplementedException(); }
        }

        public double Skewness
        {
            get { throw new NotImplementedException(); }
        }

        #endregion Properties

        #region Methods

        public double ProbabilityMassFunction(params double[] x)
        {
            throw new NotImplementedException();
        }

        public double ProbabilityDensityFunction(params double[] x)
        {
            double probability = 0;
            if (x[0] == 0)
            {
                probability = 0.4;
            }
            else if (x[0] == 1)
            {
                probability = 0.6;
            }
            return probability;

        }

        public IDistribution Evaluate(double[] observations)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[] observations, double[] weights)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, out double likelihood)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, double[] weights, out double likelihood)
        {
            throw new NotImplementedException();
        }

        #endregion Methods

        #region ICloneable Members

        public object Clone()
        {
            throw new NotImplementedException();
        }

        #endregion

        public bool Equals(IDistribution other)
        {
            var d = other as InStateSDistribution;
            if (d == null)
            {
                return false;
            }
            return (Mean - d.Mean == 0 && Variance - d.Variance == 0);
        }
    }

    public class InStateTDistribution : IDistribution
    {
        #region Variables

        public double Variance
        {
            get { throw new NotImplementedException(); }
        }

        public double Mean
        {
            get { throw new NotImplementedException(); }
        }

        public int Median
        {
            get { throw new NotImplementedException(); }
        }

        public double Skewness
        {
            get { throw new NotImplementedException(); }
        }

        #endregion Variables

        #region Methods

        public double ProbabilityMassFunction(params double[] x)
        {
            throw new NotImplementedException();
        }

        public double ProbabilityDensityFunction(params double[] x)
        {
            double probability = 0;
            if (x[0] == 0)
            {
                probability = 0.5;
            }
            else if (x[0] == 1)
            {
                probability = 0.5;
            }
            return probability;
        }

        public IDistribution Evaluate(double[] observations)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[] observations, double[] weights)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, out double likelihood)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, double[] weights, out double likelihood)
        {
            throw new NotImplementedException();
        }

        #endregion Methods

        #region ICloneable Members

        public object Clone()
        {
            throw new NotImplementedException();
        }

        #endregion

        public bool Equals(IDistribution other)
        {
            var d = other as InStateTDistribution;
            if (d == null)
            {
                return false;
            }
            return (Mean - d.Mean == 0 && Variance - d.Variance == 0);
        }
    }

    public class HealthyDistribution : IDistribution
    {
        #region Properties

        public double Variance
        {
            get { throw new NotImplementedException(); }
        }

        public double Mean
        {
            get { throw new NotImplementedException(); }
        }

        public int Median
        {
            get { throw new NotImplementedException(); }
        }

        public double Skewness
        {
            get { throw new NotImplementedException(); }
        }

        #endregion Properties

        #region Methods

        public double ProbabilityMassFunction(params double[] x)
        {
            throw new NotImplementedException();
        }

        public double ProbabilityDensityFunction(params double[] x)
        {
            double probability = 0;
            if (x[0] == 0)
            {
                probability = 0.5;
            }
            else if (x[0] == 1)
            {
                probability = 0.3;
            }
            else if (x[0] == 2)
            {
                probability = 0.2;
            }
            return probability;
        }

        public IDistribution Evaluate(double[] observations)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[] observations, double[] weights)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, out double likelihood)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, double[] weights, out double likelihood)
        {
            throw new NotImplementedException();
        }

        #endregion Methods

        #region ICloneable Members

        public object Clone()
        {
            throw new NotImplementedException();
        }

        #endregion

        public bool Equals(IDistribution other)
        {
            var d = other as HealthyDistribution;
            if (d == null)
            {
                return false;
            }
            return (Mean - d.Mean == 0 && Variance - d.Variance == 0);
        }
    }

    public class OkDistribution : IDistribution
    {
        #region Properties

        public double Variance
        {
            get { throw new NotImplementedException(); }
        }

        public double Mean
        {
            get { throw new NotImplementedException(); }
        }

        public int Median
        {
            get { throw new NotImplementedException(); }
        }

        public double Skewness
        {
            get { throw new NotImplementedException(); }
        }

        #endregion Properties

        #region Methods

        public double ProbabilityMassFunction(params double[] x)
        {
            throw new NotImplementedException();
        }

        public double ProbabilityDensityFunction(params double[] x)
        {
            double probability = 0;
            if (x[0] == 0)
            {
                probability = 0.2;
            }
            else if (x[0] == 1)
            {
                probability = 0.6;
            }
            else if (x[0] == 2)
            {
                probability = 0.2;
            }
            return probability;
        }

        public IDistribution Evaluate(double[] observations)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[] observations, double[] weights)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, out double likelihood)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, double[] weights, out double likelihood)
        {
            throw new NotImplementedException();
        }

        #endregion Properties

        #region ICloneable Members

        public object Clone()
        {
            throw new NotImplementedException();
        }

        #endregion

        public bool Equals(IDistribution other)
        {
            var d = other as OkDistribution;
            if (d == null)
            {
                return false;
            }
            return (Mean - d.Mean == 0 && Variance - d.Variance == 0);
        }
    }

    public class SickDistribution : IDistribution
    {
        #region Properties

        public double Variance
        {
            get { throw new NotImplementedException(); }
        }

        public double Mean
        {
            get { throw new NotImplementedException(); }
        }

        public int Median
        {
            get { throw new NotImplementedException(); }
        }

        public double Skewness
        {
            get { throw new NotImplementedException(); }
        }

        #endregion Properties

        #region Methods

        public double ProbabilityMassFunction(params double[] x)
        {
            throw new NotImplementedException();
        }

        public double ProbabilityDensityFunction(params double[] x)
        {
            double probability = 0;
            if (x[0] == 0)
            {
                probability = 0.2;
            }
            else if (x[0] == 1)
            {
                probability = 0.3;
            }
            else if (x[0] == 2)
            {
                probability = 0.5;
            }
            return probability;
        }

        public IDistribution Evaluate(double[] observations)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[] observations, double[] weights)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, out double likelihood)
        {
            throw new NotImplementedException();
        }

        public IDistribution Evaluate(double[][] observations, double[] weights, out double likelihood)
        {
            throw new NotImplementedException();
        }

        #endregion Properties

        #region ICloneable Members

        public object Clone()
        {
            throw new NotImplementedException();
        }

        #endregion

        public bool Equals(IDistribution other)
        {
            var d = other as SickDistribution;
            if (d == null)
            {
                return false;
            }
            return (Mean - d.Mean == 0 && Variance - d.Variance == 0);
        }
    }
}
