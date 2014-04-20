namespace HmmDotNet.MachineLearning.Algorithms.Search.Base
{
    public interface ICrossover
    {
        decimal CrossoverProbability { get; }

        IChromosome<T> RunCrossover<T>(IChromosome<T> x, IChromosome<T> y, decimal probability);
    }
}
