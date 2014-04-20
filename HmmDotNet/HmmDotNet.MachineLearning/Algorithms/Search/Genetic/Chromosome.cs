using HmmDotNet.MachineLearning.Algorithms.Search.Base;

namespace HmmDotNet.MachineLearning.Algorithms.Search
{
    public class Chromosome<T> : IChromosome<T>
    {
        public Chromosome(IGene<T>[] representation)
        {
            Representation = representation;
        }

        public object Clone()
        {
            var representation = Representation.Clone();
            return new Chromosome<T>((IGene<T>[])representation);
        }

        public IGene<T>[] Representation { get; protected set; }

        public decimal FintnessValue { get; set; }
    }
}
