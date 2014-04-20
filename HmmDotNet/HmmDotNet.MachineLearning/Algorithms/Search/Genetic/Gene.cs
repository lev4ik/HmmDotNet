using System;
using HmmDotNet.Extentions;
using HmmDotNet.MachineLearning.Algorithms.Search.Base;

namespace HmmDotNet.MachineLearning.Algorithms.Search
{
    public class Gene<T> : IGene<T>
    {
        public Gene(T[] representation, decimal multationProbability)
        {
            Representation = representation;
            MutationProbability = multationProbability;
        }

        public object Clone()
        {
            return new Gene<T>(Representation, MutationProbability);
        }

        public T[] Representation { get; private set; }
        
        public decimal MutationProbability { get; private set; }

        public void Mutate(decimal probability)
        {
            if (MutationProbability >= probability)
            {
                var rd = new Random();
                var index = rd.Next(0, Representation.Length);                
                Representation.Swap(index, index == 0 ? Representation.Length - 1 : index - 1);
            }
        }
    }
}
