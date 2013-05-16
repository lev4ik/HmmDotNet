using System.Text;
using HmmDotNet.Mathematic.Extentions;

namespace HmmDotNet.Mathematic
{
    public class Vector : IVector<double>
    {
        #region Private Variables

        private double[] _v;

        #endregion Private Variables

        #region Constructors

        public Vector(double[] v)
        {
            _v = (double[])v.Clone();
        }   

        #endregion Constructors

        public override string ToString()
        {
            var builder = new StringBuilder();
            builder.Append("{");
            for (int i = 0; i < Dimention; i++)
            {
                builder.Append(V[i]);
                if (i < Dimention - 1)
                {
                    builder.Append(",");
                }
            }
            builder.Append("}");
            return builder.ToString();
        }

        public int Dimention
        {
            get { return _v.Length; }
        }

        public double[] V
        {
            get { return _v; }
        }

        public double Product(double[] v)
        {
            return _v.Product(v);
        }

        public double[] Product(double x)
        {
            return _v.Product(x);
        }

        public double[] Add(double[] v)
        {
            return _v.Add(v);
        }

        public double[] Add(double x)
        {
            return _v.Add(x);
        }

        public double[] Substract(double[] v)
        {
            return _v.Substruct(v);
        }

        public double[] Substract(double x)
        {
            return _v.Substruct(x);
        }

        public double[,] OuterProduct(double[] v)
        {
            return _v.OuterProduct(v);
        }
    }
}
