namespace HmmDotNet.MachineLearning.Base
{
    public class ObservationWithLikelihood<T>
    {
        public double LogLikelihood { get; set; }

        public T Observation { get; set; }

        public int PlaceInSequence { get; set; }

        public int NumberOfGuesses { get; set; }
    }
}
