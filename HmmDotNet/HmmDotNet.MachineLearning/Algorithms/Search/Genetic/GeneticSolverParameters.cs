using HmmDotNet.MachineLearning.Algorithms.Search.Base;

namespace HmmDotNet.MachineLearning.Algorithms.Search.Genetic
{
    public class GeneticSolverParameters : IGeneticSolverParameters
    {
        public int NumberOfGenerations { get; set; }
        public int PopulationSize { get; set; }
        public int TournamentSize { get; set; }
        public decimal CrossOverProbability { get; set; }
        public decimal MutationProbability { get; set; }
        public int BestChromosomeTransferPercent { get; set; }
        public int RandomPopulationSize { get; set; }
    }
}
