using HmmDotNet.MachineLearning.Algorithms.VaribaleEstimationCalculator;
using HmmDotNet.Statistics.Distributions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HmmDotNet.MachineLearning.Test.Estimators
{
    [TestClass]
    public class TransitionProbabilityEstimatorTest
    {
        [TestMethod]
        public void TransitionProbabilityEstimator_ParameterPassed_TransitionProbabilityEstimatorCreated()
        {
            var N = 100;
            var T = 200;
            var alpha = new double[N][];
            var beta = new double[N][];
            var transitionProbabilityMatrix = new double[N][];
            var emissions = new IDistribution[N];
            var observations = new double[T][];
            var weights = new double[T];

            var estimator = new TransitionProbabilityEstimator<IDistribution>(alpha, beta, transitionProbabilityMatrix, emissions, observations, weights);

            Assert.IsNotNull(estimator);
        }

        [TestMethod]
        public void Estimate_NormalizedAndTransitionProbabilityMatrixNotNull_TransitionProbabilityMatrixReturned()
        {
            
        }

        [TestMethod]
        public void Estimate_NormalizedAndTransitionProbabilityMatrixNull_TransitionProbabilityMatrixCalculatedAndReturned()
        {

        }
    }
}
