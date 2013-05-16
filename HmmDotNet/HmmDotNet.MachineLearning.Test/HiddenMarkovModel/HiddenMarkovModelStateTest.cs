using HmmDotNet.Statistics.Distributions.Univariate;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Mathematic.Extentions;

namespace HmmDotNet.Logic.Test.MachineLearning
{
    [TestClass]
    public class HiddenMarkovModelStateTest
    {
        [TestMethod]
        public void HiddenMarkovModelState_NumberOfStateGreaterThanZero_ErgodicModelCreated()
        {
            var modelState = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = 4 });//new HiddenMarkovModelState<IDistribution>(4);

            Assert.AreEqual(ModelType.Ergodic, modelState.Type);
            
            Assert.AreEqual(1, modelState.TransitionProbabilityMatrix[0].Sum());
            Assert.AreEqual(1, modelState.TransitionProbabilityMatrix[1].Sum());
            Assert.AreEqual(1, modelState.TransitionProbabilityMatrix[2].Sum());
            Assert.AreEqual(1, modelState.TransitionProbabilityMatrix[3].Sum());

            Assert.AreEqual(1, modelState.Pi.Sum());
        }

        [TestMethod]
        public void HiddenMarkovModelState_NumberOfStateAndDeltaGreaterThanZero_LeftRightModelCreated()
        {
            var modelState = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { NumberOfStates = 4, Delta = 2 });//new HiddenMarkovModelState<IDistribution>(4, 2);

            Assert.AreEqual(ModelType.LeftRight, modelState.Type);

            Assert.AreEqual(1, modelState.TransitionProbabilityMatrix[0].Sum());
            Assert.AreEqual(1, modelState.TransitionProbabilityMatrix[1].Sum());
            Assert.AreEqual(1, modelState.TransitionProbabilityMatrix[2].Sum());
            Assert.AreEqual(1, modelState.TransitionProbabilityMatrix[3].Sum());

            Assert.AreEqual(1, modelState.Pi.Sum());
            Assert.AreEqual(1, modelState.Pi[0]);
        }

        [TestMethod]
        public void HiddenMarkovModelState_LeftRightModelParametersPassed_ModelTypeIsLeftRight()
        {
            var pi = new double[] {1d, 0, 0, 0};
            var tpm = new double[4][];
            tpm[0] = new double[] { 1d / 3d, 1d / 3d, 1d / 3d , 0};
            tpm[1] = new double[] { 0, 1d / 3d, 1d / 3d, 1d / 3d };
            tpm[2] = new double[] { 0, 0, 0.5, 0.5 };
            tpm[3] = new double[] { 0, 0, 0, 1d };

            var modelState = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { Pi = pi, TransitionProbabilityMatrix = tpm, Emissions = new NormalDistribution[4] });//new HiddenMarkovModelState<IDistribution>(pi, tpm, new IDistribution[4]);

            Assert.AreEqual(ModelType.LeftRight, modelState.Type);
        }

        [TestMethod]
        public void HiddenMarkovModelState_ErgodicModelParametersPassed_ModelTypeIsErgodic()
        {
            var pi = new double[] { 1d / 4d, 1d / 4d, 1d / 4d, 1d / 4d };
            var tpm = new double[4][];
            tpm[0] = new double[] { 1d / 4d, 1d / 4d, 1d / 4d, 1d / 4d };
            tpm[1] = new double[] { 1d / 4d, 1d / 4d, 1d / 4d, 1d / 4d };
            tpm[2] = new double[] { 1d / 4d, 1d / 4d, 1d / 4d, 1d / 4d };
            tpm[3] = new double[] { 1d / 4d, 1d / 4d, 1d / 4d, 1d / 4d };

            var modelState = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { Pi = pi, TransitionProbabilityMatrix = tpm, Emissions = new NormalDistribution[4] });//new HiddenMarkovModelState<IDistribution>(pi, tpm, new IDistribution[4]);

            Assert.AreEqual(ModelType.Ergodic, modelState.Type);
        }

        [TestMethod]
        public void HiddenMarkovModelState_CustomModelParametersPassed_ModelTypeIsCustom()
        {
            var pi = new double[] { 1d / 3d, 1d / 3d, 1d / 3d, 0 };
            var tpm = new double[4][];
            tpm[0] = new double[] { 1d / 3d, 1d / 3d, 1d / 3d, 0 };
            tpm[1] = new double[] { 1d / 4d, 1d / 4d, 1d / 4d, 1d / 4d };
            tpm[2] = new double[] { 1d / 3d, 1d / 3d, 1d / 3d, 0 };
            tpm[3] = new double[] { 1d / 4d, 1d / 4d, 1d / 4d, 1d / 4d };

            var modelState = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution>() { Pi = pi, TransitionProbabilityMatrix = tpm, Emissions = new NormalDistribution[4] });//new HiddenMarkovModelState<IDistribution>(pi, tpm, new IDistribution[4]);

            Assert.AreEqual(ModelType.Custom, modelState.Type);
        }
    }
}
