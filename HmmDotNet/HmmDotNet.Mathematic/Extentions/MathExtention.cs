using System;
using System.Collections.Generic;

namespace HmmDotNet.Mathematic.Extentions
{
    public static class MathExtention
    {
        #region ArgMax

        public static T ArgMax<T>(this IList<T> values, Func<T, double> f, T value, out int index) where T : IComparable, IComparable<T>
        {
            var i = 0;
            var x = f(value);
            var max = default(T);
            index = 0;
            var o = values[0];
            foreach (var y in values)
            {
                if (f(y) > x)
                {
                    max = y;
                    index = i;
                }
                i++;
            }

            return max;
        }

        #endregion ArgMax

        #region Factorial

        public static long Factorial(int k)
        {
            if (k < 0)
                return 0;
            if (k == 0) 
                return 1;

            return k * Factorial(k-1);
        }

        #endregion Factorial

        #region Binomial Coefficient

        public static long BinomialCoefficient(int k, int n)
        {
            if (n - k < 0 || k < 0 || n < 0)
                return 0;
            return Factorial(n) / (Factorial(k) * Factorial(n - k));
        }

        #endregion Binomial Coefficient

        #region Double extenetions

        public static bool EqualsTo(this double a, double b)
        {
            return !(Math.Abs(a - b) > 0);
        }

        public static bool EqualsToZero(this double a)
        {
            return (Math.Abs(a - 0.0) >= 0.0 && Math.Abs(a - 0.0) <= 0.0);
        }

        /// <summary>
        ///  sqrt(a^2 + b^2) without under/overflow.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>

        public static double Hypot(double a, double b)
        {
            double r;
            if (Math.Abs(a) > Math.Abs(b))
            {
                r = b / a;
                r = Math.Abs(a) * Math.Sqrt(1 + r * r);
            }
            else if (b != 0)
            {
                r = a / b;
                r = Math.Abs(b) * Math.Sqrt(1 + r * r);
            }
            else
            {
                r = 0.0;
            }
            return r;
        }
        #endregion Double extenetions
    }
}
