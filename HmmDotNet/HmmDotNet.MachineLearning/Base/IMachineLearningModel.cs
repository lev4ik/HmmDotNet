namespace HmmDotNet.MachineLearning
{
    public interface IMachineLearningModel
    {
        double Likelihood { get; }

        void Train(double[][] observations);
    }
}
