using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.MachineLearning;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;
using HmmDotNet.Statistics.Distributions.Univariate;

namespace HmmDotNet.Logic.Test.MachineLearning.HiddenMarkovModel
{
    [TestClass]
    public class HiddenMarkovModelFactoryTest
    {
        private int NumberOfStates = 4;

        [TestMethod]
        public void GetModel_IMultivariateDistributionParameters_IMultivariateDistributionStateCreated()
        {
            var parameters = new ModelCreationParameters<IMultivariateDistribution>() { NumberOfStates = NumberOfStates };
            var model = HiddenMarkovModelFactory.GetModel(parameters);

            Assert.IsInstanceOfType(model, typeof(HiddenMarkovModelMultivariateGaussianDistribution));
        }

        [TestMethod]
        public void GetModel_DiscreteDistributionParameters_DiscreteDistributionStateCreated()
        {
            var parameters = new ModelCreationParameters<DiscreteDistribution>() { NumberOfStates = NumberOfStates };
            var model = HiddenMarkovModelFactory.GetModel(parameters);

            Assert.IsInstanceOfType(model, typeof(HmmDotNet.MachineLearning.HiddenMarkovModel));
        }

        [TestMethod]
        public void GetModel_MixtureWithIMultivariateDistributionParameters_MixtureWithIMultivariateDistributionStateCreated()
        {
            var parameters = new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfStates = NumberOfStates };
            var model = HiddenMarkovModelFactory.GetModel(parameters);

            Assert.IsInstanceOfType(model, typeof(HiddenMarkovModelMixtureDistribution));
        }

        [TestMethod]
        public void GetModel_UnivariateAndNormalDistributionParameters_UnivariateAndNormalDistributionStateCreated()
        {
            var parameters = new ModelCreationParameters<HmmDotNet.Statistics.Distributions.Univariate.NormalDistribution>() { NumberOfStates = NumberOfStates };
            var model = HiddenMarkovModelFactory.GetModel(parameters);

            Assert.IsInstanceOfType(model, typeof(HiddenMarkovModelGaussianDistribution));
        }

    }
}
