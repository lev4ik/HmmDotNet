namespace HmmDotNet.MachineLearning.Algorithms.Search.Base
{
    public interface IMutatable
    {
        decimal MutationProbability { get; }

        IChromosome<T> RunMutation<T>(IChromosome<T> x, decimal probability);
    }
}
