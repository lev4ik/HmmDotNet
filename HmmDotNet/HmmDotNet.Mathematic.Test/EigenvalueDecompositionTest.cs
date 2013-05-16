using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Mathematic.MatrixDecomposition;

namespace HmmDotNet.Logic.Test.Mathematic
{
    [TestClass]
    public class EigenvalueDecompositionTest
    {
        [TestMethod]
        public void Calculate_4x4Matrix_Eigenvector()
        {
            var m = new double[,] {{1157.34222212636, 1036.61999992791, 987.991111034635, 1161.36666659532}, 
                                   {1036.61999992791, 1016.84666662175, 919.293333279443, 1199.55666662978}, 
                                   {987.991111034635, 919.293333279443, 856.782222162621, 1053.38666661627}, 
                                   {1161.36666659532, 1199.55666662978, 1053.38666661627, 1452.72666664381}};

            var d = new EigenvalueDecomposition();
            d.Calculate(m);

            Assert.IsNotNull(d.Eigenvectors);
            Assert.AreEqual(4, d.Eigenvectors.Length);
        }

        [TestMethod]
        public void Calculate_4x4Matrix_Eigenvalue()
        {
            var m = new double[,] {{1157.34222212636, 1036.61999992791, 987.991111034635, 1161.36666659532}, 
                                   {1036.61999992791, 1016.84666662175, 919.293333279443, 1199.55666662978}, 
                                   {987.991111034635, 919.293333279443, 856.782222162621, 1053.38666661627}, 
                                   {1161.36666659532, 1199.55666662978, 1053.38666661627, 1452.72666664381}};

            var d = new EigenvalueDecomposition();
            d.Calculate(m);

            Assert.IsNotNull(d.Eigenvalues);
            Assert.AreEqual(4, d.Eigenvalues.Length);
        }

        [TestMethod]
        public void Calculate_4x4Matrix_Test()
        {
            var m = new double[,] {{1157.34222212636, 1036.61999992791, 987.991111034635, 1161.36666659532}, 
                                   {1036.61999992791, 1016.84666662175, 919.293333279443, 1199.55666662978}, 
                                   {987.991111034635, 919.293333279443, 856.782222162621, 1053.38666661627}, 
                                   {1161.36666659532, 1199.55666662978, 1053.38666661627, 1452.72666664381}};

            var d = new EigenvalueDecomposition();
            d.Calculate(m);

            var mv = m.Product(d.Eigenvectors.Convert());
            var dv = d.Eigenvectors.Convert().Product(d.Eigenvalues.Convert());

            Assert.AreEqual(mv.Length, dv.Length);
        }
    }
}
