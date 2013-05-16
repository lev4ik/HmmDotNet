using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Mathematic.MatrixDecomposition;

namespace HmmDotNet.Logic.Test.Mathematic
{
    [TestClass]
    public class CholeskyDecompositionTest
    {
        [TestMethod]
        public void TestCholeskyDecomposition()
        {
            var m = new double[3, 3];
            m[0, 0] = 25;
            m[0, 1] = 15;
            m[0, 2] = -5;
            m[1, 0] = 15;
            m[1, 1] = 18;
            m[1, 2] = 0;
            m[2, 0] = -5;
            m[2, 1] = 0;
            m[2, 2] = 11;

            var d = new Cholesky();
            d.Calculate(m);
            Assert.AreEqual(d.Decomposition[0, 0], 5);
            Assert.AreEqual(d.Decomposition[0, 1], 3);
            Assert.AreEqual(d.Decomposition[0, 2], -1);
            Assert.AreEqual(d.Decomposition[1, 0], 3);
            Assert.AreEqual(d.Decomposition[1, 1], 3);
            Assert.AreEqual(d.Decomposition[1, 2], 1);
            Assert.AreEqual(d.Decomposition[2, 0], -1);
            Assert.AreEqual(d.Decomposition[2, 1], 1);
            Assert.AreEqual(d.Decomposition[2, 2], 3);
        }
    }
}
