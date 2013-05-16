using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HmmDotNet.Extentions.Test
{
    [TestClass]
    public class ExtentionMethodTest
    {
        [TestMethod]
        public void ArrayExtentions_Add()
        {
            var arr = new double[] {3.2, 4.5, 7, 9.0};
            var symbol = 7.67;

            var result = arr.Add(symbol);

            Assert.AreEqual(5, result.Length);
            Assert.AreEqual(symbol, result[arr.Length]);
        }
    }
}
