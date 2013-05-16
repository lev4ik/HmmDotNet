using System;

namespace HmmDotNet.Mathematic.Extentions
{
    public static class LogExtention
    {
        /// <summary>
        ///     Exponent of x. Check if x in NaN performed.
        /// </summary>
        /// <param name="x">positive number</param>
        /// <returns></returns>
        public static double eExp(double x)
        {
            if (double.IsNaN(x))
            {
                return 0;
            }
            return Math.Exp(x);
        }

        /// <summary>
        ///     Computes Logarithm of x, with ragrds if x is NaN.
        /// </summary>
        /// <param name="x">positive number</param>
        /// <returns></returns>
        public static double eLn(double x)
        {
            if (x > 0)
            {
                return Math.Log(x);
            }
            if (x == 0)
            {
                return Double.NaN;
            }
            throw new ArithmeticException("Input parameter is negative");
        }

        /// <summary>
        ///     Computes sum of two log fuction results x and y.
        /// </summary>
        /// <param name="x">eLn(x)</param>
        /// <param name="y">eLn(y)</param>
        /// <returns></returns>
        public static double eLnSum(double x, double y)
        {
            if (double.IsNaN(x) || double.IsNaN(y))
            {
                if (double.IsNaN(x))
                {
                    return y;
                }
                return x;
            }
            if (x > y)
            {
                return x + eLn(1 + Math.Exp(y - x));
            }
            return y + eLn(1 + Math.Exp(x - y));
        }

        /// <summary>
        ///     Performs product of two value x and y.
        ///     When x and y are values after Log funtion.
        /// </summary>
        /// <param name="x">eLn(x)</param>
        /// <param name="y">eLn(y)</param>
        /// <returns></returns>
        public static double eLnProduct(double x, double y)
        {
            if (double.IsNaN(x) || double.IsNaN(y))
            {
                return double.NaN;
            }
            return x + y;
        }
    }
}
