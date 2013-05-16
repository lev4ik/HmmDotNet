using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Mathematic.Extentions;

namespace HmmDotNet.Logic.Test.Mathematic
{
    [TestClass]
    public class LogExtentionsTest
    {
        [TestMethod]
        public void eExp_XisNaN_ZeroReturned()
        {
            var x = double.NaN;

            var result = LogExtention.eExp(x);

            Assert.AreEqual(0, result);
        }

        [TestMethod]
        public void eExp_XisNumber_ePowerXReturned()
        {
            var x = 1d;

            var result = LogExtention.eExp(x);

            Assert.AreEqual(Math.E, result);
        }

        [TestMethod]
        public void eLn_XisGreaterThanZero_LogXReturned()
        {
            var x = 1d;

            var result = LogExtention.eLn(x);

            Assert.AreEqual(0, result);
        }

        [TestMethod]
        public void eLn_XisZero_NaNReturned()
        {
            var x = 0;

            var result = LogExtention.eLn(x);

            Assert.AreEqual(double.NaN, result);
        }

        [TestMethod]
        public void eLnSum_XisNaN_YReturned()
        {
            var x = double.NaN;
            var y = 1d;

            var result = LogExtention.eLnSum(x, y);

            Assert.AreEqual(y, result);
        }

        [TestMethod]
        public void eLnSum_YisNaN_XReturned()
        {
            var x = 1d;
            var y = double.NaN;

            var result = LogExtention.eLnSum(x, y);

            Assert.AreEqual(x, result);
        }

        [TestMethod]
        public void eLnSum_XGreaterThanY_eLnSumReturned()
        {
            var x = 2d;
            var y = 1d;

            var result = LogExtention.eLnSum(x, y);

            Assert.AreEqual(x + LogExtention.eLn(1 + Math.Exp(y - x)), result);
        }

        [TestMethod]
        public void eLnSum_YGreaterThanX_eLnSumReturned()
        {
            var x = 1d;
            var y = 2d;

            var result = LogExtention.eLnSum(x, y);

            Assert.AreEqual(y + LogExtention.eLn(1 + Math.Exp(x - y)), result);
        }

        [TestMethod]
        public void eLnProduct_XisNaN_YReturned()
        {
            var x = double.NaN;
            var y = 1d;

            var result = LogExtention.eLnProduct(x, y);

            Assert.AreEqual(double.NaN, result);
        }

        [TestMethod]
        public void eLnProduct_YisNaN_XReturned()
        {
            var x = 1d;
            var y = double.NaN;

            var result = LogExtention.eLnProduct(x, y);

            Assert.AreEqual(double.NaN, result);
        }

        [TestMethod]
        public void eLnProduct_YandXnotEqualsNan_XPlusYReturned()
        {
            var x = 1d;
            var y = 1d;

            var result = LogExtention.eLnProduct(x, y);

            Assert.AreEqual(x + y, result);
        }
    }
}
