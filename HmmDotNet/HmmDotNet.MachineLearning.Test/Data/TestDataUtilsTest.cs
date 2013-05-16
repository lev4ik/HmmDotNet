using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HmmDotNet.Logic.Test.MachineLearning.Data
{
    [TestClass]
    public class TestDataUtilsTest
    {
        [TestMethod]
        public void TestGetSp500Data()
        {
            var util = new TestDataUtils();

            var arr = util.GetSvcData(util.Sp500FilePath, new DateTime(1950, 1, 3), new DateTime(2012, 11, 16));

            Assert.AreEqual(15822, arr.Length);
        }
    }
}
