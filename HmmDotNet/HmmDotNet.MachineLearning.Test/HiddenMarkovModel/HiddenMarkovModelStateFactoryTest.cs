using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.Estimators;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Multivariate;
using HmmDotNet.Statistics.Distributions.Univariate;

namespace HmmDotNet.Logic.Test.MachineLearning.HiddenMarkovModel
{
    [TestClass]
    public class HiddenMarkovModelStateFactoryTest
    {
        private int NumberOfStates = 4;

        [TestMethod]
        public void GetState_IMultivariateDistributionParameters_IMultivariateDistributionStateCreated()
        {
            var parameters = new ModelCreationParameters<IMultivariateDistribution>() { NumberOfStates = NumberOfStates };
            var model = HiddenMarkovModelStateFactory.GetState(parameters);

            Assert.IsInstanceOfType(model, typeof(HiddenMarkovModel<IMultivariateDistribution>));
        }

        [TestMethod]
        public void GetState_DiscreteDistributionParameters_DiscreteDistributionStateCreated()
        {
            var parameters = new ModelCreationParameters<DiscreteDistribution>() { NumberOfStates = NumberOfStates };
            var model = HiddenMarkovModelStateFactory.GetState(parameters);

            Assert.IsInstanceOfType(model, typeof(HiddenMarkovModel<DiscreteDistribution>));
        }

        [TestMethod]
        public void GetState_MixtureWithIMultivariateDistributionParameters_MixtureWithIMultivariateDistributionStateCreated()
        {
            var parameters = new ModelCreationParameters<Mixture<IMultivariateDistribution>>() { NumberOfStates = NumberOfStates };
            var model = HiddenMarkovModelStateFactory.GetState(parameters);

            Assert.IsInstanceOfType(model, typeof(HiddenMarkovModel<Mixture<IMultivariateDistribution>>));
        }

        [TestMethod]
        public void GetState_MultivariateAndNormalDistributionParameters_MultivariateAndNormalDistributionStateCreated()
        {
            var parameters = new ModelCreationParameters<HmmDotNet.Statistics.Distributions.Multivariate.NormalDistribution>() { NumberOfStates = NumberOfStates };
            var model = HiddenMarkovModelStateFactory.GetState(parameters);

            Assert.IsInstanceOfType(model, typeof(HiddenMarkovModel<HmmDotNet.Statistics.Distributions.Multivariate.NormalDistribution>));
        }

        [TestMethod]
        public void GetState_UnivariateAndNormalDistributionParameters_UnivariateAndNormalDistributionStateCreated()
        {
            var parameters = new ModelCreationParameters<HmmDotNet.Statistics.Distributions.Univariate.NormalDistribution>() { NumberOfStates = NumberOfStates };
            var model = HiddenMarkovModelStateFactory.GetState(parameters);

            Assert.IsInstanceOfType(model, typeof(HiddenMarkovModel<HmmDotNet.Statistics.Distributions.Univariate.NormalDistribution>));
        }
    }
}
