using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Mathematic.Extentions;

namespace HmmDotNet.Logic.Test.Mathematic
{
    [TestClass]
    public class MathExtentionTest
    {
        #region ArgMax

        [TestMethod]
        public void ArgMax_ListPassed_MaximumArgumentFound()
        {
            var list = new double[] {5.3, 4.3, 10, 2.2};
            int index;
            var argmax = list.ArgMax(d => { return d + 10; }, 5, out index);

            Assert.AreEqual(10, argmax);
            Assert.AreEqual(2, index);
        }

        
        #endregion ArgMax

        #region Factorial Testing



        [TestMethod]
        public void TestFactorial()
        {
            int k = 5;
            long result = MathExtention.Factorial(k);
            Assert.AreEqual(result, 120);
        }

        [TestMethod]
        public void TestFactorialNegative()
        {
            int k = -1;
            long result = MathExtention.Factorial(k);
            Assert.AreEqual(result, 0);
        }

        [TestMethod]
        public void TestFactorialZero()
        {
            int k = 0;
            long result = MathExtention.Factorial(k);
            Assert.AreEqual(result, 1);
        }

        #endregion Factorial Testing

        #region Binomial cooeficient testing

        [TestMethod]
        public void TestBinomialCooeficient()
        {
            int n = 5;
            int k = 2;
            long result = MathExtention.BinomialCoefficient(k, n);
            Assert.AreEqual(result, 10);
        }

        [TestMethod]
        public void TestBinomialCooeficientNegative()
        {
            int n = -1;
            int k = 2;
            long result = MathExtention.BinomialCoefficient(k, n);
            Assert.AreEqual(result, 0);
        }

        [TestMethod]
        public void TestBinomialCooeficientZero()
        {
            int n = 0;
            int k = 0;
            long result = MathExtention.BinomialCoefficient(k, n);
            Assert.AreEqual(result, 1);
        }

        #endregion Binomial cooeficient testing

        [TestMethod]
        public void EqualsTo_OneAndTwo_NotEquals()
        {
            var a = 1d;
            var b = 2d;

            Assert.IsFalse(a.EqualsTo(b));
        }

        [TestMethod]
        public void EqualsTo_OneAndMinusOne_NotEquals()
        {
            var a = 1d;
            var b = -1d;

            Assert.IsFalse(a.EqualsTo(b));
        }

        [TestMethod]
        public void EqualsTo_OneAndOne_Equals()
        {
            var a = 1d;
            var b = 1d;

            Assert.IsTrue(a.EqualsTo(b));
        }

        [TestMethod]
        public void EqualsToZero_One_NotEqual()
        {
            var a = 1d;

            Assert.IsFalse(a.EqualsToZero());
        }

        [TestMethod]
        public void EqualsToZero_PlusOne_NotEqual()
        {
            var a = -1d;

            Assert.IsFalse(a.EqualsToZero());
        }

        [TestMethod]
        public void EqualsToZero_Zero_Equals()
        {
            var a = 0d;

            Assert.IsTrue(a.EqualsToZero());
        }

        [TestMethod]
        public void EqualsToZero_ZeroWithFloatingPointPrecision_NotEquals()
        {
            var a = 0.0000000001;

            Assert.IsFalse(a.EqualsToZero());
        }
    }
}
